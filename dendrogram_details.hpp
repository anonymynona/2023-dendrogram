#ifndef DENDROGRAM_DETAILS_HPP
#define DENDROGRAM_DETAILS_HPP

#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp>  // iota
#include <Kokkos_Core.hpp>

struct UnweightedEdge {
  int source;
  int target;
};

constexpr int UNDEFINED_CHAIN_VALUE = -1;
constexpr int ROOT_CHAIN_VALUE = -2;
constexpr int FOLLOW_CHAIN_VALUE = -3;

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> findSmallestVertexIncidentEdges(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> edges) {
  KokkosExt::ScopedProfileRegion guard(
      "Dendrogram::find_smallest_vertex_incident_edges");

  auto const num_edges = edges.size();
  auto const num_vertices = num_edges + 1;

  // Find smallest incident edge for each vertex
  Kokkos::View<int *, MemorySpace> smallest_vertex_incident_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::smallest_vertex_incident_edges"),
      num_vertices);
  Kokkos::deep_copy(exec_space, smallest_vertex_incident_edges, INT_MAX);

  Kokkos::parallel_for(
      "Dendrogram::find_smallest_vertex_incident_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        auto &edge = edges(e);
        Kokkos::atomic_min(&smallest_vertex_incident_edges(edge.source), e);
        Kokkos::atomic_min(&smallest_vertex_incident_edges(edge.target), e);
      });

  return smallest_vertex_incident_edges;
}

// Determine alpha edges
//
// An alpha-edge is an edge that has both children as edges in the dendrogram.
// In other words, an edge that is not an alpha edge has at most one child
// edge. Assuming the edges are sorted in the order of increasing weights, an
// edge e is an alpha-edge if both vertices have an incident edge that is
// smaller than e.
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> findAlphaEdges(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> sorted_edges,
    Kokkos::View<int *, MemorySpace> smallest_vertex_incident_edges) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::alpha_edges");

  auto const num_edges = sorted_edges.size();

  Kokkos::View<int *, MemorySpace> alpha_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::alpha_edges"),
      num_edges);
  int num_alpha_edges;
  Kokkos::parallel_scan(
      "Dendrogram::determine_alpha_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e, int &update, bool final_pass) {
        auto &edge = sorted_edges(e);
        if (smallest_vertex_incident_edges(edge.source) < e &&
            smallest_vertex_incident_edges(edge.target) < e) {
          if (final_pass) alpha_edges(update) = e;
          ++update;
        }
      },
      num_alpha_edges);
  Kokkos::resize(alpha_edges, num_alpha_edges);

  return alpha_edges;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> assignAlphaVertices(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> edges,
    Kokkos::View<int *, MemorySpace> alpha_edge_indices) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::assign_alpha_vertices");

  Kokkos::View<int *, MemorySpace> alpha_inverse_map(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::inverse_map"),
      edges.size());
  Kokkos::deep_copy(exec_space, alpha_inverse_map, -1);
  Kokkos::parallel_for(
      "Dendrogram::update_inverse_map",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                          alpha_edge_indices.size()),
      KOKKOS_LAMBDA(int i) { alpha_inverse_map(alpha_edge_indices(i)) = i; });

  auto const num_edges = edges.size();
  auto num_vertices = num_edges + 1;
  [[maybe_unused]] auto num_alpha_edge_indices = (int)alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::alpha_vertices"),
      num_vertices);

  {
    // Do initial union-find on the subgraphs

    ArborX::iota(exec_space, alpha_vertices);

    ArborX::Details::UnionFind<MemorySpace> union_find(alpha_vertices);
    Kokkos::parallel_for(
        "Dendrogram::alpha_vertices_union_find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
        KOKKOS_LAMBDA(int e) {
          if (alpha_inverse_map(e) == -1) {
            // Not an alpha edge, merge edge vertices
            auto &edge = edges(e);
            union_find.merge(edge.source, edge.target);
          }
        });
    // finalize union-find
    Kokkos::parallel_for(
        "Dendrogram::finalize_union-find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int const i) {
          // ##### ECL license (see LICENSE.ECL) #####
          int next;
          int vstat = alpha_vertices(i);
          int const old = vstat;
          while (vstat > (next = alpha_vertices(vstat))) {
            vstat = next;
          }
          if (vstat != old) alpha_vertices(i) = vstat;
        });
  }

  {
    // Map found alpha-vertices back to [0, #alpha vertices) range

    Kokkos::View<int *, MemorySpace> map(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "Dendrogram::map_back"),
        num_vertices);
    Kokkos::deep_copy(exec_space, map, -1);

    Kokkos::parallel_for(
        "Dendrogram::find_unique_entries",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          map(alpha_vertices(i)) = 1;
        });
    int num_unique_entries = 0;
    Kokkos::parallel_scan(
        "Dendrogram::map_scan",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
          if (map(i) != -1) {
            if (is_final) map(i) = partial_sum;
            ++partial_sum;
          }
        },
        num_unique_entries);
    assert(num_unique_entries == num_alpha_edge_indices + 1);

    Kokkos::parallel_for(
        "Dendrogram::remap_alpha_vertices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          alpha_vertices(i) = map(alpha_vertices(i));
        });
  }

  return alpha_vertices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<UnweightedEdge *, MemorySpace> buildAlphaMST(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> edges,
    Kokkos::View<int *, MemorySpace> alpha_edge_indices,
    Kokkos::View<int *, MemorySpace> alpha_vertices) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::build_alpha_mst");

  auto const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<UnweightedEdge *, MemorySpace> alpha_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::alpha_mst_edges"),
      num_alpha_edges);
  Kokkos::parallel_for(
      "Dendrogram::build_alpha_mst_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int k) {
        auto &edge = edges(alpha_edge_indices(k));
        alpha_edges(k) = {alpha_vertices(edge.source),
                          alpha_vertices(edge.target)};
      });

  return alpha_edges;
}

template <typename ExecutionSpace, typename MemorySpace>
void buildAlphaIncidenceMatrix(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> edges,
    Kokkos::View<int *, MemorySpace> alpha_edge_indices,
    Kokkos::View<int *, MemorySpace> alpha_vertices,
    Kokkos::View<int *, MemorySpace> &alpha_mat_offsets,
    Kokkos::View<int *, MemorySpace> &alpha_mat_edges) {
  KokkosExt::ScopedProfileRegion guard(
      "Dendrogram::build_alpha_incidence_matrix");

  // FIXME we could use sort to construct the incidence matrix,
  // this should be faster

  auto num_alpha_edges = alpha_edge_indices.size();
  auto num_vertices = num_alpha_edges + 1;

  Kokkos::realloc(alpha_mat_offsets, num_vertices + 1);
  Kokkos::parallel_for(
      "Dendrogram::compute_alpha_mat_counts",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int const ee) {
        auto const &edge = edges(alpha_edge_indices(ee));

        auto const i = alpha_vertices(edge.source);
        auto const j = alpha_vertices(edge.target);

        Kokkos::atomic_increment(&alpha_mat_offsets(i));
        Kokkos::atomic_increment(&alpha_mat_offsets(j));
      });
  ArborX::exclusivePrefixSum(exec_space, alpha_mat_offsets);

  assert(KokkosExt::lastElement(exec_space, alpha_mat_offsets) ==
         2 * (int)num_alpha_edges);

  KokkosExt::reallocWithoutInitializing(
      exec_space, alpha_mat_edges,
      KokkosExt::lastElement(exec_space, alpha_mat_offsets));

  // FIXME: if we can make it sorted, we could improve computing sided parents
  auto offsets = KokkosExt::clone(exec_space, alpha_mat_offsets);
  Kokkos::parallel_for(
      "Dendrogram::compute_alpha_mat_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int const ee) {
        auto const e = alpha_edge_indices(ee);  // original edge index
        auto const &edge = edges(e);

        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.source)), 1)) = e;
        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.target)), 1)) = e;
      });
}

template <typename ExecutionSpace, typename MemorySpace>
void updateSidedParents(ExecutionSpace const &exec_space,
                        Kokkos::View<UnweightedEdge const *, MemorySpace> edges,
                        int largest_alpha_index,
                        Kokkos::View<int *, MemorySpace> alpha_vertices,
                        Kokkos::View<int *, MemorySpace> &alpha_mat_offsets,
                        Kokkos::View<int *, MemorySpace> &alpha_mat_edges,
                        Kokkos::View<int *, MemorySpace> global_map,
                        Kokkos::View<int *, MemorySpace> &sided_level_parents) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::update_sided_parents");

  Kokkos::parallel_for(
      "Dendrogram::update_sided_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);

        auto alpha_vertex = alpha_vertices(edge.source);
        if (alpha_vertices(edge.target) != alpha_vertex) {
          // This is an alpha-edge, skip
          return;
        }

        if (e > largest_alpha_index) {
          // No alpha-ancestors, assign to the root chain
          sided_level_parents(global_map(e)) = ROOT_CHAIN_VALUE;
          return;
        }

        int largest_smaller = -1;
        int smallest_larger = INT_MAX;
        for (int k = alpha_mat_offsets(alpha_vertex);
             k < alpha_mat_offsets(alpha_vertex + 1); ++k) {
          auto alpha_e = alpha_mat_edges(k);
          if (alpha_e < e && alpha_e > largest_smaller)
            largest_smaller = alpha_e;
          if (alpha_e > e && alpha_e < smallest_larger)
            smallest_larger = alpha_e;
        }
        assert(largest_smaller != INT_MAX || smallest_larger != -1);

        if (largest_smaller == -1 && smallest_larger != INT_MAX) {
          // No smaller incident alpha-edge.
          // Can immediately assign the parent.
          auto const &alpha_edge = edges(smallest_larger);

          bool is_left_side =
              (alpha_vertices(alpha_edge.source) == alpha_vertex);
          sided_level_parents(global_map(e)) =
              2 * global_map(smallest_larger) + static_cast<int>(is_left_side);
        } else {
          sided_level_parents(global_map(e)) =
              FOLLOW_CHAIN_VALUE - global_map(largest_smaller);
        }
      });
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> compressGlobalMap(
    ExecutionSpace const &exec_space,
    Kokkos::View<int *, MemorySpace> global_map,
    Kokkos::View<int *, MemorySpace> alpha_edge_indices) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::compress_global_map");

  auto const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> compressed_global_map(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::global_map"),
      num_alpha_edges);

  Kokkos::parallel_for(
      "Dendrogram::find_compression_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int i) {
        compressed_global_map(i) = global_map(alpha_edge_indices(i));
      });

  return compressed_global_map;
}

template <typename ExecutionSpace, typename MemorySpace, typename Parents>
void computeParents(ExecutionSpace const &exec_space,
                    Kokkos::View<int *, MemorySpace> sided_level_parents,
                    Parents &parents) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::compute_edge_parents");

  auto num_edges = sided_level_parents.size();

  // Encode both sided parent and edge into long long
  // This way, once we sort based on this value, edges with the same sided
  // parent will already be sorted in increasing order.
  Kokkos::View<long long *, MemorySpace> keys(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "Dendrogram::keys"),
      num_edges);

  constexpr int shift = 32;

  Kokkos::parallel_for(
      "Dendrogram::compute_sided_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        long long key = sided_level_parents(e);
        if (key <= FOLLOW_CHAIN_VALUE) {
          int next = FOLLOW_CHAIN_VALUE - key;
          do {
            key = sided_level_parents(next);
            if (key <= FOLLOW_CHAIN_VALUE)
              next = FOLLOW_CHAIN_VALUE - key;
            else if (key >= 0) {
              next = key / 2;
              if (next > e) break;
            } else if (key == ROOT_CHAIN_VALUE)
              break;
          } while (true);
        }
        if (key == ROOT_CHAIN_VALUE) key = INT_MAX;

        keys(e) = (key << shift) + e;
      });

  auto permute = ArborX::Details::sortObjects(exec_space, keys);

  Kokkos::parallel_for(
      "Dendrogram::compute_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const i) {
        int e = permute(i);
        if (i == (int)num_edges - 1)
          parents(e) = -1;
        else if ((keys(i) >> shift) == (keys(i + 1) >> shift))
          parents(e) = permute(i + 1);
        else
          parents(e) = (keys(i) >> shift) / 2;
      });
}

template <typename ExecutionSpace, typename MemorySpace, typename Parents>
void assignVertexParents(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> sorted_edges,
    Kokkos::View<int *, MemorySpace> smallest_vertex_incident_edges,
    Parents &parents) {
  KokkosExt::ScopedProfileRegion guard("Dendrogram::assign_vertex_parents");

  auto const num_edges = sorted_edges.size();
  auto const vertices_offset = num_edges;

  Kokkos::parallel_for(
      "Dendrogram::assign_vertex_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        auto const &edge = sorted_edges(e);
        if (smallest_vertex_incident_edges(edge.source) == e)
          parents(vertices_offset + edge.source) = e;
        if (smallest_vertex_incident_edges(edge.target) == e)
          parents(vertices_offset + edge.target) = e;
      });
}

#endif

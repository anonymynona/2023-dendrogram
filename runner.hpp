#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <ArborX_MinimumSpanningTree.hpp>
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fstream>

#include "dendrogram.hpp"
#include "parameters.hpp"

template <typename ExecutionSpace, typename Primitives>
auto runner(ExecutionSpace const &exec_space, Primitives const &primitives,
            int core_min_size) {
  Kokkos::Profiling::pushRegion("runner");

  using MemorySpace = typename Primitives::memory_space;

  Kokkos::Profiling::pushRegion("mst");
  ArborX::Details::MinimumSpanningTree<MemorySpace> mst(exec_space, primitives,
                                                        core_min_size);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dendrogram");
  Dendrogram<MemorySpace> dendrogram(exec_space, mst.edges);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return dendrogram;
}

template <int DIM>
auto loadData(std::string const &filename) {
  using ArborX::ExperimentalHyperGeometry::Point;

  std::cout << "Reading in \"" << filename << "\"";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  assert(input.good());

  std::vector<Point<DIM>> v;

  int num_points = 0;
  int dim = 0;
  input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
  input.read(reinterpret_cast<char *>(&dim), sizeof(int));

  assert(dim == DIM);

  v.resize(num_points);
  // Directly read into a point
  input.read(reinterpret_cast<char *>(v.data()),
             num_points * sizeof(Point<DIM>));
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "") {
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

template <int DIM>
void run(Parameters const &params) {
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto data = loadData<DIM>(params.filename);

  auto const primitives = vec2view<MemorySpace>(data, "primitives");

  Kokkos::Profiling::pushRegion("total");
  auto dendrogram = runner(exec_space, primitives, params.core_min_size);
  Kokkos::Profiling::popRegion();
}

#endif

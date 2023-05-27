#include <ArborX_Version.hpp>
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <fstream>

#include "parameters.hpp"
#include "runner.hpp"

// FIXME: ideally, this function would be next to `loadData` in
// dbscan_timpl.hpp. However, that file is used for explicit instantiation,
// which would result in multiple duplicate symbols. So it is kept here.
int getDataDimension(std::string const &filename) {
  std::ifstream input(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Error reading file \"" + filename + "\"");

  int num_points;
  int dim;
  input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
  input.read(reinterpret_cast<char *>(&dim), sizeof(int));

  input.close();

  return dim;
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  namespace bpo = boost::program_options;

  Parameters params;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(1), "minpts")
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc << '\n';
    return EXIT_SUCCESS;
  }

  // Print out the runtime parameters
  printf("minpts            : %d\n", params.core_min_size);
  printf("filename          : %s\n", params.filename.c_str());

  assert(!params.filename.empty());
  int dim = getDataDimension(params.filename);

  switch (dim) {
    case 2:
      run<2>(params);
      break;
    case 3:
      run<3>(params);
      break;
    case 4:
      run<4>(params);
      break;
    case 5:
      run<5>(params);
      break;
    case 6:
      run<6>(params);
      break;
    default:
      std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
  }

  return EXIT_SUCCESS;
}

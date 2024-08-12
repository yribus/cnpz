#include <catch_amalgamated.hpp>
#include <spdlog/spdlog.h>

#include "cnpz.h"

using namespace cnpz;

// TODO: provide in-memory backend for these tests

TEST_CASE("Simple ZIP", "[host]")
{
    // Can use unzip -v to check
    NpzFile npz("ziptest.zip");
    spdlog::info("Creating file {}", npz.full_path());
    std::string data = "Words are loud\n";
    npz.add_file("bubble.txt", data);
    npz.close();
    CHECK(npz.num_files() == 1);

    // Read back the file as string
    std::ifstream ifs(npz.full_path(), std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    CHECK(content.size() == 133);
    CHECK(content.find("Words are loud\n") == 40);
}
//
// TEST_CASE("Simple NPZ", "[host]")
// {
//     NpzFile npz("npztest");
//     spdlog::info("Creating file {}", npz.full_path());
//     auto matrix = Tensor2f::ones({{3, 2}});
//     size_t filesize = npz.add_array("matrix", matrix.data(), matrix.shape_vector());
//     npz.close();
//     CHECK(npz.num_files() == 1);
//     CHECK(filesize == 152);
//
//     // Read back the file as string
//     std::ifstream ifs(npz.full_path(), std::ios::binary);
//     std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
//     CHECK(content.size() == 270);
// }
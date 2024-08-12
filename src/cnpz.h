#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>

namespace cnpz {
    using shape_type = std::vector<size_t>;

    enum class CompressionMethod : uint16_t {
        STORED = 0,
        DEFLATE = 8
    };

    // Template-based mapping of C++ types to NumPy descriptors
    template<typename T> std::string numpy_descr();

    class NpzFile {
    public:
        // if filename does not have .npz or .zip extension, we add .npz
        explicit NpzFile(const std::string& filename);
        ~NpzFile();
        void close();

        std::string full_path() const;
        inline std::string filename() const { return filename_; }
        inline int num_files() const { return num_entries_; }

        // We need to pass header separately so convenient to support 2 buffers
        // Returns number of bytes written for the file itself (after compression)
        size_t add_file_from_buffers(
            const std::string& name, const char* buf0, size_t size0, const char* buf1, size_t size1,
            time_t timestamp = 0, CompressionMethod compression = CompressionMethod::STORED);
        void add_file(const std::string& name, const char* data, size_t size, time_t timestamp = 0){
            add_file_from_buffers(name, data, size, nullptr, 0, timestamp);
        }
        inline void add_file(const std::string &name, const std::string& file_data, time_t timestamp = 0) {
            add_file(name, file_data.c_str(), file_data.size(), timestamp);
        }

        template<typename T>
        size_t add_array(const std::string& name, const T* data, const shape_type& shape, time_t timestamp = 0) {
            std::string type_descr = numpy_descr<T>();
            return add_array_of_type(name, type_descr, sizeof(T), reinterpret_cast<const char*>(data), shape, timestamp);
        }
    private:
        // Returns number of bytes written. Adds .npy extension to name if missing
        size_t add_array_of_type(const std::string& name, const std::string& type_descr, size_t type_size,
                                 const char* data, const shape_type& shape, time_t timestamp = 0);

        static std::string create_npy_header(const std::string &type_descr, const shape_type& shape);

        std::string filename_;
        std::ofstream fs_;
        std::ostringstream central_dir_;
        uint16_t num_entries_{0};
    };
}
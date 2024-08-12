#include "cnpz.h"
#include <zlib.h>
#include <cassert>
#include <ctime>
#include <vector>
#include <numeric>
#include <filesystem>

using namespace cnpz;
using std::string;
namespace fs = std::filesystem;

#define PACKED_STRUCT struct __attribute__((packed))

// These struct rely on little-endian byte order

const uint32_t ZIP_LOCAL_FILE_HEADER_SIG = 0x04034b50;  //PK\3\4
const uint32_t ZIP_CENTRAL_DIRECTORY_FILE_HEADER_SIG = 0x02014b50;  //PK\1\2
const uint16_t VERSION_MADE_BY = 20;  // Everyone uses 20

// Zip local file header
// Structure is:
//   uint32_t ZIP_LOCAL_FILE_HEADER_SIG
//   fields from ZipLocalFileHeader
//   filename
//   extra field
PACKED_STRUCT ZipLocalFileHeader {
    uint16_t version_needed_to_extract{VERSION_MADE_BY};
    uint16_t general_purpose_bit_flag{0};
    uint16_t compression_method{static_cast<uint16_t>(CompressionMethod::STORED)};
    uint16_t last_mod_file_time{0};
    uint16_t last_mod_file_date{0};
    uint32_t crc32{0};
    uint32_t compressed_size{0};
    uint32_t uncompressed_size{0};
    uint16_t filename_length{0};
    uint16_t extra_field_length{0};
};

// Zip central directory file header
// Structure is:
//   uint32_t ZIP_CENTRAL_DIRECTORY_FILE_HEADER_SIG
//   uint16_t version_made_by{20};
//   fields from ZipLocalFileHeader
//   fields from ZipCentralDirectoryFileHeaderSuffix
//   filename
//   extra field
PACKED_STRUCT ZipCentralDirectoryFileHeaderSuffix {
    uint16_t file_comment_length{0};
    uint16_t disk_number_start{0};
    uint16_t internal_file_attr{0};
    uint32_t external_file_attr{0};
    uint32_t relative_offset_of_local_header{0};
};

// Zip end of central directory record
PACKED_STRUCT ZipEndOfCentralDirectoryRecord {
    uint32_t signature{0x06054b50};
    uint16_t disk_number{0};
    uint16_t central_directory_disk_number{0};
    uint16_t num_entries_on_disk{0};
    uint16_t num_entries_total{0};
    uint32_t central_directory_size{0};
    uint32_t central_directory_offset{0};
    uint16_t comment_size{0};
};

// Helper functions

string filename_with_extension(const string& filename, const string& extension)
{
    return filename.ends_with(extension) ? filename : filename + extension;
}

// These assume little-endian
inline void write2(std::ostream& os, uint16_t value)
{
    os.write(reinterpret_cast<const char*>(&value), 2);
}

inline void write4(std::ostream& os, uint32_t value)
{
    os.write(reinterpret_cast<const char*>(&value), 4);
}

// NPY
const uint32_t NPY_ARRAY_ALIGN = 64;  // Seems at some point NPY switched from 16 to 64

PACKED_STRUCT NpyHeader {
    char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};   // File name (8 bytes)
    char major_version = 1;  // Major version
    char minor_version = 0;  // Minor version
    uint16_t header_len = 0;   // uint32_t in NPY 2.0 (vs uint16_t in NPY 1.0)
};

namespace cnpz {
    template<> std::string numpy_descr<int8_t>() { return "|i1"; }
    template<> std::string numpy_descr<int16_t>() { return "<i2"; }
    template<> std::string numpy_descr<int32_t>() { return "<i4"; }
    template<> std::string numpy_descr<int64_t>() { return "<i8"; }
    template<> std::string numpy_descr<uint8_t>() { return "|u1"; }
    template<> std::string numpy_descr<uint16_t>() { return "<u2"; }
    template<> std::string numpy_descr<uint32_t>() { return "<u4"; }
    template<> std::string numpy_descr<uint64_t>() { return "<u8"; }
    template<> std::string numpy_descr<float>() { return "<f4"; }
    template<> std::string numpy_descr<double>() { return "<f8"; }
    template<> std::string numpy_descr<char>() { return "|S1"; }
    template<> std::string numpy_descr<std::complex<float>>() { return "<c8"; }
    template<> std::string numpy_descr<std::complex<double>>() { return "<c16"; }
}

// NpzFile
NpzFile::NpzFile(const string& filename)
:
    filename_{filename.ends_with(".zip") ? filename : filename_with_extension(filename, ".npz")},
    fs_{filename_, std::ios::binary}
{
    if (!fs_.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }
}

NpzFile::~NpzFile()
{
    close();
}

std::string NpzFile::full_path() const
{
    // Return absolute path
    return fs::absolute(filename_).string();
}

void NpzFile::close()
{
    string central_dir = central_dir_.str();
    uint32_t central_dir_offset = fs_.tellp();

    ZipEndOfCentralDirectoryRecord eocd;
    eocd.num_entries_on_disk = num_entries_;
    eocd.num_entries_total = num_entries_;
    eocd.central_directory_size = central_dir.size();
    eocd.central_directory_offset = central_dir_offset;
    fs_.write(reinterpret_cast<const char*>(central_dir.c_str()), central_dir.size());
    fs_.write(reinterpret_cast<const char*>(&eocd), sizeof(eocd));
    // TODO: support comments
    fs_.close();
}

void compress_chunk(z_stream *p_strm, const Bytef *source, size_t source_len, Bytef* dest, size_t dest_len, bool finish)
{
    p_strm->next_out = dest;
    p_strm->avail_out = dest_len;
    p_strm->next_in = (z_const Bytef *)source;
    p_strm->avail_in = source_len;

    int ret = deflate(p_strm, finish ? Z_FINISH : Z_NO_FLUSH);

    if (finish) {
        assert(ret == Z_STREAM_END);
    } else {
        assert(ret == Z_OK || ret == Z_BUF_ERROR);
    }
}


size_t NpzFile::add_file_from_buffers(const string& name,
                                    const char* buf0, size_t size0,
                                    const char* buf1, size_t size1,
                                    time_t timestamp, CompressionMethod compression)
{
    bool use_crc = false;
    num_entries_++;
    if (name.size() > 0xffff) {
        throw std::runtime_error("Filename too long: " + name);
    }
    uint32_t filename_length = name.size();

    ZipLocalFileHeader local_header;
    if (timestamp == 0) {
        timestamp = std::time(nullptr); // use current time
    }
    struct tm* utctm = gmtime(&timestamp);
    local_header.last_mod_file_time = (utctm->tm_hour << 11) + (utctm->tm_min << 5) + (utctm->tm_sec/2);
    local_header.last_mod_file_date = ((utctm->tm_year-80) << 9) + ((utctm->tm_mon+1) << 5) + utctm->tm_mday;

    uint32_t crc = 0u;
    if (use_crc) {
        crc32(crc, reinterpret_cast<const Bytef *>(buf0), size0);
        if (buf1) { crc = crc32(crc, reinterpret_cast<const Bytef *>(buf1), size1); }
    }
    local_header.crc32 = crc;

    uint32_t total_size = size0 + (buf1 ? size1 : 0);
    local_header.filename_length = filename_length;
    local_header.uncompressed_size = total_size;

    std::vector<unsigned char> compressed;
    if (compression == CompressionMethod::DEFLATE) {
        local_header.compression_method = static_cast<uint16_t>(compression);

        z_stream strm;
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;
        // Have to use window of -15 to get raw deflate format
        int ret = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 9, Z_DEFAULT_STRATEGY);
        if (ret != Z_OK) {
            throw std::runtime_error("Failed to initialize zlib");
        }

        // Compress
        compressed.resize(compressBound(total_size));
        compress_chunk(&strm, reinterpret_cast<const Bytef*>(buf0), size0,
                       reinterpret_cast<Bytef*>(compressed.data()), compressed.size(), !buf1);
        assert(strm.avail_in == 0);
        size_t compressed_size = strm.total_out;

        if (buf1) {
            compress_chunk(&strm, reinterpret_cast<const Bytef*>(buf1), size1,
                           reinterpret_cast<Bytef*>(compressed.data()) + compressed_size, compressed.size() - compressed_size, true);
            compressed_size = strm.total_out;
        }
        deflateEnd(&strm);
        compressed.resize(compressed_size);
        local_header.compressed_size = compressed_size;
        assert(strm.total_in == total_size);
    } else {
        local_header.compressed_size = total_size;
    }

    // Write local header
    uint32_t local_header_offset = fs_.tellp();
    write4(fs_, ZIP_LOCAL_FILE_HEADER_SIG);
    fs_.write(reinterpret_cast<const char*>(&local_header), sizeof(local_header));
    fs_.write(name.c_str(), filename_length);
    if (compression == CompressionMethod::DEFLATE) {
        fs_.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    } else {
        fs_.write(buf0, size0);
        if (buf1) { fs_.write(buf1, size1); }
    }

    // Add to central directory
    write4(central_dir_, ZIP_CENTRAL_DIRECTORY_FILE_HEADER_SIG);
    write2(central_dir_, VERSION_MADE_BY);
    central_dir_.write(reinterpret_cast<const char*>(&local_header), sizeof(local_header));
    ZipCentralDirectoryFileHeaderSuffix suffix;
    // TOCONSIDER: suffix.external_file_attr = 0640 << 16 for example (high 16 bits are OS permissions)
    suffix.relative_offset_of_local_header = local_header_offset;
    central_dir_.write(reinterpret_cast<const char*>(&suffix), sizeof(ZipCentralDirectoryFileHeaderSuffix));
    central_dir_.write(name.c_str(), filename_length);

    return local_header.compressed_size;
}

size_t NpzFile::add_array_of_type(const std::string& name, const std::string& type_descr, size_t type_size,
                                  const char* data, const shape_type& shape, time_t timestamp)
{
    std::string npy_header = create_npy_header(type_descr, shape);
    string full_name = filename_with_extension(name, ".npy");
    size_t data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * type_size;
    return add_file_from_buffers(full_name, npy_header.c_str(), npy_header.size(), data, data_size, timestamp);
}

std::string NpzFile::create_npy_header(const string& type_descr, const shape_type& shape)
{
    assert(shape.size() > 0);

    NpyHeader npy_header;
    npy_header.header_len = NPY_ARRAY_ALIGN - sizeof(NpyHeader);
    std::string header;
    header.reserve(128); // Usually be of size 64
    header.append(reinterpret_cast<const char*>(&npy_header), sizeof(npy_header));
    header += "{'descr': '";
    header += type_descr;
    header += "', 'fortran_order': False, 'shape': (";
    header += std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
        header += ",";
        header += std::to_string(shape[i]);
    }
    if (shape.size() == 1) header += ",";  // Python tuple
    header += ")}";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int pad_count = NPY_ARRAY_ALIGN - (header.size() + 1) % NPY_ARRAY_ALIGN;  // 1 more for newline
    header.append(pad_count, ' ');
    header.append("\n");
    assert(header.size() <= 0xffff); // Would need to switch to NPY 2.0 otherwise
    if (header.size() > NPY_ARRAY_ALIGN) {
        npy_header.header_len = header.size() - sizeof(NpyHeader);
        header.replace(8, 2, reinterpret_cast<const char*>(&npy_header.header_len), 2);
    }
    assert(header.size() % NPY_ARRAY_ALIGN == 0);
    return header;
}

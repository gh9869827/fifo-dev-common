#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#if defined(FIFO_DEV_COMMON_ENABLE_BOOST)
#  include <boost/asio.hpp>
#elif defined(__has_include)
#  if __has_include(<boost/asio.hpp>)
#    include <boost/asio.hpp>
#    define FIFO_DEV_COMMON_ENABLE_BOOST 1
#  endif
#endif

/**
 * @brief FifoBuffer: A thin convenience wrapper around a dynamically growing contiguous buffer for
 * cross-language serialization
 * 
 * This class provides a unified serialization/deserialization interface that matches the format 
 * implemented in Python (FifoSerializable). It enables seamless data exchange between Python and
 * C++ by maintaining identical binary formats for primitive types, containers, and complex objects.
 * 
 * All serialization and deserialization is performed in little-endian byte order, regardless of 
 * host architecture. This ensures cross-platform and cross-language compatibility for all primitive
 * and composite types.
 * 
 * Key Features:
 * - Cross-language compatibility with Python FifoSerializable format
 * - Type-safe serialization with compile-time checks
 * - Support for primitive types, strings, arrays, optional values, and nested objects
 * - Automatic capacity management with efficient memory usage
 * - Length-prefixed writing for variable-size data structures
 * 
 * Serialization Format:
 * - Primitive types: Direct binary representation (little-endian)
 * - Strings: uint32_t length + UTF-8 bytes
 * - Arrays: uint32_t count + elements
 * - Optional values: uint8_t presence flag + value (if present)
 * - Nested objects: Recursive serialization via serialize() method
 * 
 * Usage Pattern:
 * Complex objects should implement a serialize(FifoBuffer& buf) method for writing
 * and a constructor taking FifoBuffer& for reading. This enables automatic code
 * generation from Python definitions to create matching C++ serializable classes.
 * 
 * Thread Safety:
 * This class is NOT thread-safe. External synchronization is required for concurrent access.
 */
/**
 * @brief Lightweight UUID passthrough used for serialization.
 *
 * The structure intentionally leaves its storage uninitialized when default-constructed so that
 * callers that immediately deserialize into the buffer avoid an unnecessary zero-fill. Callers that
 * require a known all-zero UUID can use the @ref nil factory.
 */
struct FifoUuid {
    /**
     * @brief Raw 16-byte UUID storage.
     *
     * The bytes are left in an unspecified state after default construction; they are meant to be
     * written by serialization routines. Use @ref nil when a zero-initialized identifier is
     * required.
     */
    std::array<std::uint8_t, 16> bytes;

    /**
     * @brief Creates a canonical nil UUID (all bytes set to zero).
     */
    static constexpr FifoUuid nil() { return FifoUuid{{}}; }

    /**
     * @brief Provides read-only access to the raw UUID bytes.
     * @return Pointer to the first byte of the UUID.
     */
    constexpr const std::uint8_t* data() const { return bytes.data(); }

    /**
     * @brief Provides mutable access to the raw UUID bytes.
     * @return Pointer to the first byte of the UUID.
     */
    constexpr std::uint8_t* data() { return bytes.data(); }

    /**
     * @brief Equality comparison based on the underlying byte representation.
     */
    bool operator==(const FifoUuid&) const = default;
};

class FifoBuffer {
private:

    /**
     * @brief Underlying buffer storage.
     * Maintains contiguous storage for both read and write operations while
     * supporting dynamic growth on platforms without Boost.
     */
    std::vector<std::uint8_t> _buffer;
    std::size_t _read_pos{0};

public:
    /**
     * @brief Constructs a FifoBuffer with specified initial capacity
     * @param initial_capacity Initial buffer capacity in bytes (default: 256)
     */
    explicit FifoBuffer(std::size_t initial_capacity = 256);
    
    /**
     * @brief Clears the buffer content while preserving capacity
     */
    void clear();

    /**
     * @brief Returns the number of unread bytes remaining in the buffer
     * @return Unread byte count
     */
    std::size_t unread_bytes_count() const;
    
    /**
     * @brief Returns the total allocated capacity of the buffer
     * @return Buffer capacity in bytes
     */
    std::size_t capacity() const;

    /**
     * @brief Ensures that the buffer can accommodate additional bytes without reallocating during
     *        subsequent writes.
     * @param additional Number of additional bytes required
     */
    void ensure_capacity(std::size_t additional);

    /**
     * @brief Advances the write cursor without initializing bytes.
     *        Useful when a caller needs to back-fill a reserved area later.
     * @param n Number of bytes to append as uninitialized space
     */
    void advance_write_head(std::size_t n);

    /**
     * @brief Discards bytes from the readable portion of the buffer.
     * @param n Number of bytes to drop from the read cursor
     * @return true if successful, false if not enough data available
     */
    bool advance_read_head(std::size_t n);
    
    // ******************
    //     WRITERS       *
    // ******************

    /** @brief Writes a char value to the buffer */
    void write_char(char v);
    /** @brief Writes an int8_t value to the buffer */
    void write_int8_t(int8_t v);
    /** @brief Writes a uint8_t value to the buffer */
    void write_uint8_t(uint8_t v);
    /** @brief Writes an int16_t value to the buffer */
    void write_int16_t(int16_t v);
    /** @brief Writes a uint16_t value to the buffer */
    void write_uint16_t(uint16_t v);
    /** @brief Writes an int32_t value to the buffer */
    void write_int32_t(int32_t v);
    /** @brief Writes a uint32_t value to the buffer */
    void write_uint32_t(uint32_t v);
    /** @brief Writes an int64_t value to the buffer */
    void write_int64_t(int64_t v);
    /** @brief Writes a uint64_t value to the buffer */
    void write_uint64_t(uint64_t v);
    /** @brief Writes a float value to the buffer */
    void write_float(float v);
    /** @brief Writes a double value to the buffer */
    void write_double(double v);

    /**
     * @brief Writes a UUID to the buffer (16 bytes)
     * @param u The UUID to write
     */
    void write_uuid(const FifoUuid& u);
    
    /**
     * @brief Writes a string with length prefix (uint32_t length + UTF-8 bytes)
     * @param s The string to write
     */
    void write_string(const std::string& s);

    /**
     * @brief Writes an array of primitive integers with count prefix
     * @tparam T Must be a primitive integer type
     * @param vec The vector of integers to write
     */
    template<typename T>
    void write_array_primitive_integer(const std::vector<T>& vec);

    /**
     * @brief Writes a nested object by calling its serialize method
     * @tparam T Object type must have serialize(FifoBuffer&) method
     * @param obj The object to serialize
     */
    template<typename T>
    void write_nested_object(const T& obj);
    
    /**
     * @brief Writes an optional nested object with presence flag
     * @tparam T Object type must have serialize(FifoBuffer&) method
     * @param obj The optional object to serialize
     */
    template<typename T>
    void write_optional_nested_object(const std::optional<T>& obj);

    /**
     * @brief Writes a vector of nested objects with count prefix
     * @tparam T Object type must have serialize(FifoBuffer&) method
     * @param vec The vector of objects to serialize
     */
    template<typename T>
    void write_vector_nested_objects(const std::vector<T>& vec);

    /** @brief Writes an optional int8_t with presence flag */
    void write_optional_int8_t(const std::optional<int8_t>& v);
    /** @brief Writes an optional uint8_t with presence flag */
    void write_optional_uint8_t(const std::optional<uint8_t>& v);
    /** @brief Writes an optional int16_t with presence flag */
    void write_optional_int16_t(const std::optional<int16_t>& v);
    /** @brief Writes an optional uint16_t with presence flag */
    void write_optional_uint16_t(const std::optional<uint16_t>& v);
    /** @brief Writes an optional int32_t with presence flag */
    void write_optional_int32_t(const std::optional<int32_t>& v);
    /** @brief Writes an optional uint32_t with presence flag */
    void write_optional_uint32_t(const std::optional<uint32_t>& v);
    /** @brief Writes an optional int64_t with presence flag */
    void write_optional_int64_t(const std::optional<int64_t>& v);
    /** @brief Writes an optional uint64_t with presence flag */
    void write_optional_uint64_t(const std::optional<uint64_t>& v);
    /** @brief Writes an optional float with presence flag */
    void write_optional_float(const std::optional<float>& v);
    /** @brief Writes an optional double with presence flag */
    void write_optional_double(const std::optional<double>& v);
    /** @brief Writes an optional string with presence flag and length prefix */
    void write_optional_string(const std::optional<std::string>& s);

    /**
     * @brief Writes raw bytes to the buffer
     * @param src Pointer to source data
     * @param n Number of bytes to write
     */
    void write_bytes(const void* src, std::size_t n);

    /**
     * @brief Writes a uint32_t value at a specific position (overwrites existing data)
     * @param pos Position in buffer to write to
     * @param value Value to write
     * @throws std::runtime_error if position is out of bounds
     */
    void write_uint32_t_at(std::size_t pos, uint32_t value);

    // ******************
    //     READERS       *
    // ******************

    /**
     * @brief Reads an int8_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_int8_t(int8_t& v);
    
    /**
     * @brief Reads a uint8_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_uint8_t(uint8_t& v);
    
    /**
     * @brief Reads an int16_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_int16_t(int16_t& v);
    
    /**
     * @brief Reads a uint16_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_uint16_t(uint16_t& v);
    
    /**
     * @brief Reads an int32_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_int32_t(int32_t& v);
    
    /**
     * @brief Reads a uint32_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_uint32_t(uint32_t& v);
    
    /**
     * @brief Reads an int64_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_int64_t(int64_t& v);
    
    /**
     * @brief Reads a uint64_t value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_uint64_t(uint64_t& v);
    
    /**
     * @brief Reads a float value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_float(float& v);
    
    /**
     * @brief Reads a double value from the buffer
     * @param v Output variable
     * @return true if successful
     */
    bool read_double(double& v);

    /**
     * @brief Reads a UUID from the buffer (16 bytes)
     * @param u Output UUID variable
     * @return true if successful, false if not enough data
     */
    bool read_uuid(FifoUuid& u);
    
    /**
     * @brief Reads a string with length prefix from the buffer
     * @param out Output string variable
     * @return true if successful, false if not enough data
     */
    bool read_string(std::string& out);
    
    /**
     * @brief Reads an array of primitive integers with count prefix
     * @tparam T Must be a primitive integer type
     * @param out Output vector to store the integers
     * @return true if successful, false if not enough data
     */
    template<typename T>
    bool read_array_primitive_integer(std::vector<T>& out);

    /**
     * @brief Reads a nested object by calling its constructor with this buffer
     * @tparam T Object type must have constructor taking FifoBuffer&
     * @return The deserialized object
     * @throws std::runtime_error if deserialization fails
     */
    template<typename T>
    T read_nested_object();
    
    /**
     * @brief Reads an optional nested object with presence flag
     * @tparam T Object type must have constructor taking FifoBuffer&
     * @return std::optional containing the object if present, std::nullopt otherwise
     * @throws std::runtime_error if deserialization fails
     */
    template<typename T>
    std::optional<T> read_optional_nested_object();
    
    /**
     * @brief Reads a vector of nested objects with count prefix
     * @tparam T Object type must have constructor taking FifoBuffer&
     * @param out Output vector to store the objects
     * @return true if successful, false if deserialization fails
     */
    template<typename T>
    bool read_vector_nested_objects(std::vector<T>& out);

    /**
     * @brief Reads an optional int8_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_int8_t(std::optional<int8_t>& out);
    
    /**
     * @brief Reads an optional uint8_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_uint8_t(std::optional<uint8_t>& out);
    
    /**
     * @brief Reads an optional int16_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_int16_t(std::optional<int16_t>& out);
    
    /**
     * @brief Reads an optional uint16_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_uint16_t(std::optional<uint16_t>& out);
    
    /**
     * @brief Reads an optional int32_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_int32_t(std::optional<int32_t>& out);
    
    /**
     * @brief Reads an optional uint32_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_uint32_t(std::optional<uint32_t>& out);
    
    /**
     * @brief Reads an optional int64_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_int64_t(std::optional<int64_t>& out);
    
    /**
     * @brief Reads an optional uint64_t with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_uint64_t(std::optional<uint64_t>& out);
    
    /**
     * @brief Reads an optional float with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_float(std::optional<float>& out);
    
    /**
     * @brief Reads an optional double with presence flag
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_double(std::optional<double>& out);
    
    /**
     * @brief Reads an optional string with presence flag and length prefix
     * @param out Output variable
     * @return true if successful
     */
    bool read_optional_string(std::optional<std::string>& out);

    /**
     * @brief Reads raw bytes from the buffer
     * @param dst Destination buffer for the data
     * @param n Number of bytes to read
     * @return true if successful, false if not enough data available
     */
    bool read_bytes(void* dst, std::size_t n);

#if defined(FIFO_DEV_COMMON_ENABLE_BOOST)
    // ******************
    //     SOCKETS       *
    // ******************

    /**
     * @brief Reads exactly nb_bytes from socket into buffer (blocking)
     * @param socket The TCP socket to read from
     * @param nb_bytes Number of bytes to read
     * @return true if successful, false if socket error occurred
     */
    bool append_from_socket(boost::asio::ip::tcp::socket& socket, std::size_t nb_bytes);

    /**
     * @brief Sends entire buffer content to socket (blocking)
     * @param socket The TCP socket to send to
     * @return true if successful, false if socket error occurred
     */
    bool send_to_socket_exact(boost::asio::ip::tcp::socket& socket);
#endif

private:

    void compact();
    void consume(std::size_t n);

    // Generic optional POD writer
    template<typename T>
    void write_optional_pod(const std::optional<T>& v);

    template<typename T>
    bool read_optional_pod(std::optional<T>& out);

    const uint8_t* buffer_unread_data() const;

    template<typename T>
    void write_pod(const T& v);

    template<typename T>
    bool read_pod(T& out);

    template<class T>
    static constexpr T host_to_little(T v);

    template<class T>
    static constexpr T host_to_big(T v);

    template<class T>
    static constexpr T little_to_host(T v);

    template<class T>
    static constexpr T big_to_host(T v);

    template<class T>
    static constexpr T byteswap_any(T v);
};

// ==================
//   CONSTRUCTORS    
// ==================

inline FifoBuffer::FifoBuffer(std::size_t initial_capacity) {
    if (initial_capacity) {
        _buffer.reserve(initial_capacity);
    }
}

inline void FifoBuffer::clear() {
    _buffer.clear();
    _read_pos = 0;
}

inline std::size_t FifoBuffer::unread_bytes_count() const {
    return _buffer.size() - _read_pos;
}

inline std::size_t FifoBuffer::capacity() const {
    return _buffer.capacity();
}

inline void FifoBuffer::ensure_capacity(std::size_t additional) {
    if (additional == 0) {
        return;
    }
    if (_buffer.size() + additional <= _buffer.capacity()) {
        return;
    }
    if (_read_pos > 0) {
        compact();
        if (_buffer.size() + additional <= _buffer.capacity()) {
            return;
        }
    }
    _buffer.reserve(_buffer.size() + additional);
}

inline void FifoBuffer::advance_write_head(std::size_t n) {
    if (n == 0) {
        return;
    }
    ensure_capacity(n);
    const auto old_size = _buffer.size();
    _buffer.resize(old_size + n);
}

inline bool FifoBuffer::advance_read_head(std::size_t n) {
    if (unread_bytes_count() < n) {
        return false;
    }
    consume(n);
    return true;
}

// ==================
//     WRITERS       
// ==================

inline void FifoBuffer::write_char(char v) { write_pod(v); }
inline void FifoBuffer::write_int8_t(int8_t v) { write_pod(v); }
inline void FifoBuffer::write_uint8_t(uint8_t v) { write_pod(v); }
inline void FifoBuffer::write_int16_t(int16_t v) { write_pod(v); }
inline void FifoBuffer::write_uint16_t(uint16_t v) { write_pod(v); }
inline void FifoBuffer::write_int32_t(int32_t v) { write_pod(v); }
inline void FifoBuffer::write_uint32_t(uint32_t v) { write_pod(v); }
inline void FifoBuffer::write_int64_t(int64_t v) { write_pod(v); }
inline void FifoBuffer::write_uint64_t(uint64_t v) { write_pod(v); }
inline void FifoBuffer::write_float(float v) { write_pod(v); }
inline void FifoBuffer::write_double(double v) { write_pod(v); }

inline void FifoBuffer::write_uuid(const FifoUuid& u) {
    write_bytes(u.data(), 16);
}

inline void FifoBuffer::write_string(const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    write_uint32_t(len);
    if (len) {
        write_bytes(s.data(), len);
    }
}

template<typename T>
void FifoBuffer::write_array_primitive_integer(const std::vector<T>& vec) {
    static_assert(std::is_integral_v<T>, "T must be a primitive integer type");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    
    const uint32_t len = static_cast<uint32_t>(vec.size());
    write_uint32_t(len);

    if (len == 0) {
        return;
    }

    if constexpr (sizeof(T) > 1) {
        if constexpr (std::endian::native != std::endian::little) {
            for (const auto& v : vec) {
                const T le = host_to_little(v);  // swaps on big-endian, no-op otherwise
                write_bytes(&le, sizeof(T));
            }
            return;
        }
    }

    // Little-endian host or 1-byte elements: bulk write
    write_bytes(vec.data(), len * sizeof(T));
}

template<typename T>
void FifoBuffer::write_nested_object(const T& obj) {
    obj.serialize(*this);
}

template<typename T>
void FifoBuffer::write_optional_nested_object(const std::optional<T>& obj) {
    if (!obj) {
        write_uint8_t(0);
        return;
    }
    write_uint8_t(1);
    obj->serialize(*this);
}

template<typename T>
void FifoBuffer::write_vector_nested_objects(const std::vector<T>& vec) {
    uint32_t len = static_cast<uint32_t>(vec.size());
    write_uint32_t(len);
    for (const auto& item : vec) {
        write_nested_object(item);
    }
}

inline void FifoBuffer::write_optional_int8_t(const std::optional<int8_t>& v){ write_optional_pod(v);}    
inline void FifoBuffer::write_optional_uint8_t(const std::optional<uint8_t>& v){ write_optional_pod(v);}  
inline void FifoBuffer::write_optional_int16_t(const std::optional<int16_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_uint16_t(const std::optional<uint16_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_int32_t(const std::optional<int32_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_uint32_t(const std::optional<uint32_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_int64_t(const std::optional<int64_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_uint64_t(const std::optional<uint64_t>& v){ write_optional_pod(v);} 
inline void FifoBuffer::write_optional_float(const std::optional<float>& v){ write_optional_pod(v);}     
inline void FifoBuffer::write_optional_double(const std::optional<double>& v){ write_optional_pod(v);}

inline void FifoBuffer::write_optional_string(const std::optional<std::string>& s) {
    if (!s) {
        write_uint8_t(0);
        return;
    }
    write_uint8_t(1);
    uint32_t len = static_cast<uint32_t>(s->size());
    write_uint32_t(len);
    if (len) {
        write_bytes(s->data(), len);
    }
}

inline void FifoBuffer::write_bytes(const void* src, std::size_t n) {
    if (n == 0) {
        return;
    }
    ensure_capacity(n);
    const auto old_size = _buffer.size();
    _buffer.resize(old_size + n);
    std::memmove(_buffer.data() + old_size, src, n);
}

inline void FifoBuffer::write_uint32_t_at(std::size_t pos, uint32_t value) {
    if (pos + sizeof(uint32_t) > _buffer.size()) {
        throw std::runtime_error("write_uint32_t_at: position out of bounds");
    }

    // Convert to little-endian for consistency with write_uint32_t()
    const uint32_t le = host_to_little(value);

    std::memcpy(_buffer.data() + pos, &le, sizeof(le));
}

// ==================
//     READERS       
// ==================

inline bool FifoBuffer::read_int8_t(int8_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_uint8_t(uint8_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_int16_t(int16_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_uint16_t(uint16_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_int32_t(int32_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_uint32_t(uint32_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_int64_t(int64_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_uint64_t(uint64_t& v) { return read_pod(v); }
inline bool FifoBuffer::read_float(float& v) { return read_pod(v); }
inline bool FifoBuffer::read_double(double& v) { return read_pod(v); }

inline bool FifoBuffer::read_uuid(FifoUuid& u) {
    if (unread_bytes_count() < 16) {
        return false;
    }
    std::memcpy(u.data(), buffer_unread_data(), 16);
    consume(16);
    return true;
}

inline bool FifoBuffer::read_string(std::string& out) {
    uint32_t len;
    if (!read_uint32_t(len) || unread_bytes_count() < len) {
        return false;
    }
    if (len == 0) {
        out.clear();
        return true;
    }
    out.assign(reinterpret_cast<const char*>(buffer_unread_data()), len);
    consume(len);
    return true;
}

template<typename T>
bool FifoBuffer::read_array_primitive_integer(std::vector<T>& out) {
    static_assert(std::is_integral_v<T>, "T must be a primitive integer type");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    
    uint32_t len;
    if (!read_uint32_t(len) || unread_bytes_count() < len * sizeof(T)) {
        return false;
    }
    out.resize(len);
    if (len > 0) {
        std::memcpy(out.data(), buffer_unread_data(), len * sizeof(T));
        consume(len * sizeof(T));

        if constexpr (sizeof(T) > 1) {
            if constexpr (std::endian::native != std::endian::little) {
                for (auto& v : out) {
                    v = little_to_host<T>(v);
                }
            }
        }
    }
    return true;
}

template<typename T>
T FifoBuffer::read_nested_object() {
    return T(*this);  // Will throw std::runtime_error if deserialization fails
}

template<typename T>
std::optional<T> FifoBuffer::read_optional_nested_object() {
    uint8_t presence;
    if (!read_uint8_t(presence)) {
        throw std::runtime_error("Failed to read presence flag");
    }
    if (presence == 0) {
        return std::nullopt;
    }
    return T(*this);  // Will throw std::runtime_error if deserialization fails
}

template<typename T>
bool FifoBuffer::read_vector_nested_objects(std::vector<T>& out) {
    uint32_t len; if (!read_uint32_t(len)) return false; out.clear(); out.reserve(len);
    try {
        for (uint32_t i = 0; i < len; ++i) {
            out.emplace_back(read_nested_object<T>());
        }
    } catch (const std::runtime_error&) { return false; }
    return true;
}

inline bool FifoBuffer::read_optional_int8_t(std::optional<int8_t>& out){ return read_optional_pod(out);}    
inline bool FifoBuffer::read_optional_uint8_t(std::optional<uint8_t>& out){ return read_optional_pod(out);}  
inline bool FifoBuffer::read_optional_int16_t(std::optional<int16_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_uint16_t(std::optional<uint16_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_int32_t(std::optional<int32_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_uint32_t(std::optional<uint32_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_int64_t(std::optional<int64_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_uint64_t(std::optional<uint64_t>& out){ return read_optional_pod(out);} 
inline bool FifoBuffer::read_optional_float(std::optional<float>& out){ return read_optional_pod(out);}     
inline bool FifoBuffer::read_optional_double(std::optional<double>& out){ return read_optional_pod(out);}

inline bool FifoBuffer::read_optional_string(std::optional<std::string>& out) {
    uint8_t presence;
    
    if (!read_uint8_t(presence)) {
        return false;
    }
    if (presence == 0) {
        out.reset();
        return true;
    }
    uint32_t len;
    if (!read_uint32_t(len) || unread_bytes_count() < len) {
        return false;
    }
    if (len == 0) {
        out = std::string();
        return true;
    }
    out.emplace(reinterpret_cast<const char*>(buffer_unread_data()), len);
    consume(len);
    return true;
}

inline bool FifoBuffer::read_bytes(void* dst, std::size_t n) {
    if (unread_bytes_count() < n) {
        return false;
    }
    std::memcpy(dst, buffer_unread_data(), n);
    consume(n);
    return true;
}

#if defined(FIFO_DEV_COMMON_ENABLE_BOOST)

inline bool FifoBuffer::append_from_socket(boost::asio::ip::tcp::socket& socket, std::size_t nb_bytes) {
    if (nb_bytes == 0) {
        return true;
    }
    boost::system::error_code ec;
    ensure_capacity(nb_bytes);
    const auto start = _buffer.size();
    _buffer.resize(start + nb_bytes);
    auto* dest = _buffer.data() + start;
    const std::size_t n = boost::asio::read(
        socket,
        boost::asio::buffer(dest, nb_bytes),
        boost::asio::transfer_exactly(nb_bytes),
        ec);
    if (ec) {
        _buffer.resize(start);
        return false;
    }
    if (n < nb_bytes) {
        _buffer.resize(start + n);
        return false;
    }
    return true;
}

inline bool FifoBuffer::send_to_socket_exact(boost::asio::ip::tcp::socket& socket) {
    if (unread_bytes_count() == 0) {
        return true;
    }
    boost::system::error_code ec;
    boost::asio::write(socket, boost::asio::buffer(buffer_unread_data(), unread_bytes_count()), ec);
    return !ec;
}

#endif

// ==================
//   PRIVATE HELPERS
// ==================

template<typename T>
void FifoBuffer::write_optional_pod(const std::optional<T>& v) {
    if (!v) {
        write_uint8_t(0);
        return;
    }
    write_uint8_t(1);
    write_pod(*v);
}

template<typename T>
bool FifoBuffer::read_optional_pod(std::optional<T>& out) {
    uint8_t p;
    if (!read_uint8_t(p)) {
        return false;
    }
    if (!p) {
        out.reset();
        return true;
    }
    T tmp;
    if (!read_pod(tmp)) {
        return false;
    }
    out = tmp;
    return true;
}

inline void FifoBuffer::compact() {
    if (_read_pos == 0) {
        return;
    }
    if (_read_pos >= _buffer.size()) {
        _buffer.clear();
        _read_pos = 0;
        return;
    }
    const std::size_t unread = _buffer.size() - _read_pos;
    std::memmove(_buffer.data(), _buffer.data() + _read_pos, unread); // memmove tolerates overlapping ranges
    _buffer.resize(unread); // resize preserves capacity, so no reallocation occurs here
    _read_pos = 0;
}

inline void FifoBuffer::consume(std::size_t n) {
    _read_pos += n;
    if (_read_pos >= _buffer.size()) {
        _buffer.clear();
        _read_pos = 0;
    }
}

inline const uint8_t* FifoBuffer::buffer_unread_data() const {
    if (unread_bytes_count() == 0) {
        return nullptr;
    }
    return _buffer.data() + _read_pos;
}

template<typename T>
inline void FifoBuffer::write_pod(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    const T le = host_to_little(v);
    write_bytes(&le, sizeof(le));
}

template<typename T>
inline bool FifoBuffer::read_pod(T& out) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    if (unread_bytes_count() < sizeof(T)) {
        return false;
    }

    T raw;
    std::memcpy(&raw, buffer_unread_data(), sizeof(T));
    consume(sizeof(T));

    out = little_to_host(raw);

    return true;
}

// =======================
// Endian convertion
// =======================

template<class T>
constexpr T FifoBuffer::host_to_little(T v) {
    if constexpr (std::endian::native == std::endian::little) {
        return v;
    } else {
        return byteswap_any(v);
    }
}

template<class T>
constexpr T FifoBuffer::host_to_big(T v) {
    if constexpr (std::endian::native == std::endian::big) {
        return v;
    } else {
        return byteswap_any(v);
    }
}

template<class T>
constexpr T FifoBuffer::little_to_host(T v) {
    return host_to_little(v);
}

template<class T>
constexpr T FifoBuffer::big_to_host(T v) {
    return host_to_big(v);
}

template<class T>
constexpr T FifoBuffer::byteswap_any(T v) {
    static_assert(std::is_trivially_copyable_v<T>, "byteswap_any requires trivially copyable type");
    if constexpr (sizeof(T) == 1) {
        return v;
    } else if constexpr (sizeof(T) == 2) {
        return std::bit_cast<T>(
            std::byteswap(std::bit_cast<std::uint16_t>(v)));
    } else if constexpr (sizeof(T) == 4) {
        return std::bit_cast<T>(
            std::byteswap(std::bit_cast<std::uint32_t>(v)));
    } else if constexpr (sizeof(T) == 8) {
        return std::bit_cast<T>(
            std::byteswap(std::bit_cast<std::uint64_t>(v)));
    } else {
        static_assert(sizeof(T) <= 8, "Unsupported type size for byteswap_any");
    }
}
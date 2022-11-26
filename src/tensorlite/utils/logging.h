/**
 * \file  logging.h
 * \brief A light-weight logging module contains macros
 *        similar to glog. Most of these are from dmlc-core
 */

#ifndef TENSORLITE_UTILS_LOGGING_H_
#define TENSORLITE_UTILS_LOGGING_H_

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#define UTILS_NO_INLINE __declspec(noinline)
#else
#define UTILS_NO_INLINE __attribute__((noinline))
#endif

namespace tl {
namespace utils {
/**
 * \brief exception class that will be thrown by
 *        default logger.
 */
struct Error : std::runtime_error {
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};

class DateLogger {
public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char *HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm *pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value); // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", pnow->tm_hour,
             pnow->tm_min, pnow->tm_sec);
#endif
    return buffer_;
  }

private:
  char buffer_[9];
};

/**
 * \brief Message logger used for fatal exception
 */
class LogMessageFatal {
public:
  LogMessageFatal(const char *file, int line) { GetEntry().Init(file, line); }
  std::ostringstream &stream() { return GetEntry().log_stream; }
  // destructor does not throw by default
  UTILS_NO_INLINE ~LogMessageFatal() noexcept(false) {
    throw GetEntry().Finalize();
    abort();
  }

private:
  struct Entry {
    std::ostringstream log_stream;
    UTILS_NO_INLINE void Init(const char *file, int line) {
      DateLogger date;
      log_stream.str("");
      log_stream.clear();
      log_stream << "[" << date.HumanDate() << "]" << file << ":" << line
                 << ":";
    }
    Error Finalize() { return utils::Error(log_stream.str()); }
    // Due to a bug in MinGW, objects with non-trivial destructor cannot be
    // thread-local. See https://sourceforge.net/p/mingw-w64/bugs/527/ Hence,
    // don't use thread-local for the log stream if the compiler is MinGW.
#if !(defined(__MINGW32__) || defined(__MINGW64__))
    UTILS_NO_INLINE static Entry &ThreadLocal() {
      static thread_local Entry result;
      return result;
    }
#endif
  };
  LogMessageFatal(const LogMessageFatal &);
  void operator=(const LogMessageFatal &);

#if defined(__MINGW32__) || defined(__MINGW64__)
  UTILS_NO_INLINE Entry &GetEntry() { return entry_; }

  Entry entry_;
#else
  UTILS_NO_INLINE Entry &GetEntry() { return Entry::ThreadLocal(); }
#endif
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream &) {}
};

} // namespace utils
} // namespace tl

#define CHECK(x)                                                               \
  if (!(x))                                                                    \
  ::tl::utils::LogMessageFatal(__FILE__, __LINE__).stream()                    \
      << "Check failed: " #x << ":"
#define CHECK_EQ(x, y) CHECK(x == y)
#define CHECK_NE(x, y) CHECK(x != y)
#define CHECK_LT(x, y) CHECK(x < y)
#define CHECK_LE(x, y) CHECK(x <= y)
#define CHECK_GT(x, y) CHECK(x > y)
#define CHECK_GE(x, y) CHECK(x >= y)
#define CHECK_NOTNULL(x)                                                       \
  ((x) == NULL ? ::tl::utils::LogMessageFatal(__FILE__, __LINE__).stream()     \
                     << "Check  notnull: " #x << ' ',                          \
   (x) : (x)) // NOLINT(*)

#define LOG(level)                                                             \
  std::cerr << "[LOG " #level << "]: File: " << __FILE__                       \
            << ", Line: " << __LINE__ << ": "

#define LOG_INFO LOG(INFO)
#define LOG_ERROR LOG(ERROR)
#define LOG_WARNING LOG(WARN)
#define LOG_FATAL                                                              \
  ::tl::utils::LogMessageFatal(__FILE__, __LINE__).stream()                    \
      << "[LOG FATAL]: " // NOLINT(*)
#define LOG_QFATAL LOG_FATAL

#endif // TENSORLITE_UTILS_LOGGING_H_

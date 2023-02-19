#ifndef TENSORLITE_UTILS_SINGLETON_H_
#define TENSORLITE_UTILS_SINGLETON_H_

namespace tl {

/**
 * \brief A meta class used to generate thread-scope Meyer's singleton.
 *
 * \tparam Derived The actual singleton type.
 */
template <typename Derived> class ThreadLocalSingleton {
public:
  static Derived &GetSingleton() {
    thread_local Derived obj;
    return obj;
  }
};

/**
 * \brief A meta class used to generate global-scope Meyer's singleton.
 *
 * \tparam Derived The actual singleton type.
 */
template <typename Derived> class GlobalSingleton {
public:
  static Derived &GetSingleton() {
    static Derived obj;
    return obj;
  }
};

} // namespace tl

#endif // TENSORLITE_UTILS_SINGLETON_H_

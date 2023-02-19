#ifndef TENSORLITE_UTILS_RANDOM_H_
#define TENSORLITE_UTILS_RANDOM_H_

#include "tensorlite/float16.h"
#include "tensorlite/utils/logging.h"
#include "tensorlite/utils/singleton.h"
#include "tensorlite/utils/thread.h"
#include <random>

namespace tl {

class RandomEngine {
  using RandomEngineImpl = std::default_random_engine;

public:
  RandomEngine() {
    std::random_device rd;
    SetSeed(rd());
  }

  RandomEngine(RandomEngineImpl::result_type seed) { SetSeed(seed); }

  void SetSeed(RandomEngineImpl::result_type seed) {
    gen_.seed(seed + GetThreadId());
  }

  /*!
   * \brief Generate an arbitrary random 32-bit integer.
   */
  int32_t RandInt32() { return static_cast<int32_t>(gen_()); }

  /*!
   * \brief Generate a uniform random integer in [0, upper)
   */
  template <typename T> T RandInt(T upper) { return RandInt<T>(0, upper); }

  /*!
   * \brief Generate a uniform random integer in [lower, upper)
   */
  template <typename T> T RandInt(T lower, T upper) {
    CHECK_LT(lower, upper);
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(gen_);
  }

  /*!
   * \brief Generate a uniform random float in [0, 1)
   */
  template <typename T> T Uniform() { return Uniform<T>(0., 1.); }

  /*!
   * \brief Generate a uniform random float in [lower, upper)
   */
  template <typename T> T Uniform(T lower, T upper) {
    // Although the result is in [lower, upper), we allow lower == upper as in
    // www.cplusplus.com/reference/random/uniform_real_distribution/uniform_real_distribution/
    CHECK_LE(lower, upper);
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(gen_);
  }

  template <> fp16_t Uniform(fp16_t lower, fp16_t upper) {
    float f_lower = static_cast<float>(lower),
          f_upper = static_cast<float>(upper);
    CHECK_LE(f_lower, f_upper);
    std::uniform_real_distribution<float> dist(f_lower, f_upper);
    return static_cast<fp16_t>(dist(gen_));
  }

  /*!
   * \brief Generate a random float from standard normal distribution
   */
  template <typename T> T Normal() { return Normal<T>(0., 1.); }

  /*!
   * \brief Generate a random float from N(mean, stddev)
   */
  template <typename T> T Normal(T mean, T stddev) {
    std::normal_distribution<T> dist(mean, stddev);
    return dist(gen_);
  }

  template <> fp16_t Normal(fp16_t mean, fp16_t stddev) {
    float f_mean = static_cast<float>(mean),
          f_stddev = static_cast<float>(stddev);
    std::normal_distribution<float> dist(f_mean, f_stddev);
    return static_cast<fp16_t>(dist(gen_));
  }

  static RandomEngine &ThreadLocal() {
    return ThreadLocalSingleton<RandomEngine>::GetSingleton();
  }

private:
  RandomEngineImpl gen_;
};

} // namespace tl

#endif // TENSORLITE_UTILS_RANDOM_H_

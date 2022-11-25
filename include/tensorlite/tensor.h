#ifndef TENSORLITE_TENSOR_H_
#define TENSORLITE_TENSOR_H_

#include "tensorlite/buffer.h"
#include "tensorlite/dtype.h"

// tl for tensorlite
namespace tl {

class TensorShape {};

/**
 * \brief Tensor class
 *
 */
class Tensor {
public:
  /**
   * \brief Construct a new Tensor object
   *
   */
  Tensor();

  /**
   * \brief Construct a new Tensor object
   *
   * \param other
   */
  Tensor(const Tensor &other);

  /**
   * \brief Construct a new Tensor object
   *
   * \param other
   */
  Tensor(Tensor &&other);

  /**
   * \brief
   *
   * \param other
   * \return Tensor&
   */
  Tensor &operator=(const Tensor &other);

  /**
   * \brief
   *
   * \param other
   * \return Tensor&
   */
  Tensor &operator=(Tensor &&other);

  /**
   * \brief Destroy the Tensor object
   *
   */
  ~Tensor();

private:
  //
  DataType dtype_;

  //
  TensorShape shape_;

  //
  Buffer buffer_;
};

} // namespace tl

#endif // TENSORLITE_TENSOR_H_

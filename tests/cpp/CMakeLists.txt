file(GLOB CPP_TEST_SOURCE_LIST "${CMAKE_SOURCE_DIR}/tests/cpp/*.cc")

foreach(test_src ${CPP_TEST_SOURCE_LIST})
  string(REGEX REPLACE "${CMAKE_SOURCE_DIR}/tests/cpp/(.*)\.cc" "\\1" testname ${test_src})
  message(STATUS "[UnitTest]: Add testcase \"${testname}\"")
  add_executable(${testname} "${test_src}")
  add_dependencies(${testname} tensorlite)
  target_link_directories(${testname} PUBLIC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                                      PUBLIC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  target_link_libraries(${testname} tensorlite GTest::gtest_main)
  target_include_directories(${testname} PUBLIC "${CMAKE_SOURCE_DIR}/include"
                                                "${gtest_SOURCE_DIR}/include"
                                                "${gtest_SOURCE_DIR}")
  # add_test(NAME ${testname} COMMAND ${testname})
  gtest_discover_tests(${testname})
endforeach()

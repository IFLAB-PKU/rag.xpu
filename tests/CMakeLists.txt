if (POWERSERVE_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE powerserve CLI11::CLI11)
endif()

if (POWERSERVE_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    find_package(Threads REQUIRED)
    if (OpenCL_FOUND)
        add_executable(layer_diff_test "layer_diff_test.cpp")
        add_executable(logits_cmp "logits_cmp.cpp")
        add_executable(opencl_bandwidth_test "opencl_bandwidth_test.cpp")

        target_link_libraries(layer_diff_test PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        target_link_libraries(logits_cmp PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        target_link_libraries(opencl_bandwidth_test PRIVATE
            powerserve
            OpenCL::OpenCL
            Threads::Threads
        )

        if (NOT MSVC)
            target_compile_options(layer_diff_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(logits_cmp PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(opencl_bandwidth_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
        endif()

        target_include_directories(layer_diff_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(logits_cmp PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(opencl_bandwidth_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )

        add_custom_target(run_layer_diff_test
            COMMAND POWERSERVE_USE_OPENCL=1 ./layer_diff_test
            DEPENDS layer_diff_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running layer diff tests..."
        )

        add_custom_target(run_logits_cmp
            COMMAND POWERSERVE_USE_OPENCL=1 ./logits_cmp
            DEPENDS logits_cmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running logits compare tests..."
        )

        add_custom_target(run_opencl_bandwidth_test
            COMMAND ${CMAKE_COMMAND} -E env POWERSERVE_USE_OPENCL=1 $<TARGET_FILE:opencl_bandwidth_test>
            DEPENDS opencl_bandwidth_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running OpenCL bandwidth test..."
        )
    endif()
endif()


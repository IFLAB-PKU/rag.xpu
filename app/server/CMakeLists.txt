add_library(server.simple STATIC simple_server.cpp)
set(POWERSERVE_FAISS_DIR ${PROJECT_SOURCE_DIR}/third_party/faiss)
set(POWERSERVE_FAISS_INCLUDE_DIR ${POWERSERVE_FAISS_DIR}/include)
set(POWERSERVE_FAISS_LIB ${POWERSERVE_FAISS_DIR}/lib/android-aarch64/libfaiss.so)

if (NOT EXISTS ${POWERSERVE_FAISS_INCLUDE_DIR}/faiss/IndexFlat.h)
    message(FATAL_ERROR "Missing FAISS headers at ${POWERSERVE_FAISS_INCLUDE_DIR}. Please vendor FAISS into third_party/faiss/include.")
endif ()
if (NOT EXISTS ${POWERSERVE_FAISS_LIB})
    message(FATAL_ERROR "Missing FAISS library at ${POWERSERVE_FAISS_LIB}. Please vendor libfaiss.so into third_party/faiss/lib/android-aarch64.")
endif ()

target_include_directories(server.simple PUBLIC
    ${POWERSERVE_FAISS_INCLUDE_DIR}
)
target_link_libraries(server.simple PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
    ${POWERSERVE_FAISS_LIB}
)
if (POWERSERVE_SERVER_MULTIMODEL)
    target_compile_definitions(server.simple PUBLIC POWERSERVE_SERVER_MULTIMODEL)
endif ()

add_library(server.local STATIC local_server.cpp)
target_link_libraries(server.local PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
)
if (POWERSERVE_SERVER_MULTIMODEL)
    target_compile_definitions(server.local PUBLIC POWERSERVE_SERVER_MULTIMODEL)
endif ()

add_executable(server server.cpp)
target_link_libraries(server PRIVATE
    powerserve
    app.common
    server.simple
)

powerserve_add_binary(server)

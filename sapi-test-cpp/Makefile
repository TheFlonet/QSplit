CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wpedantic -Werror # -Wextra
LIB_PATH = -I./dwave/external

EXE_SAPI = sapi
LIBS_SAPI = -lcurl -pthread

all: sapi

sapi: main_sapi.cpp dwave/external/base64/base64.cpp
	$(CXX) $^ $(CXXFLAGS) -o $(EXE_SAPI) $(LIB_PATH) $(LIBS_SAPI)

clean:
	rm -f $(EXE_SAPI)

.PHONY: all clean

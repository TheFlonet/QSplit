CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wpedantic -Werror # -Wextra
LIB_PATH = -I./qsplit/external -I./dwave/external

EXE_SAPI = build/sapi
LIBS_SAPI = -lcurl -pthread

EXE_QUBO = build/qubo

all: sapi qubo

sapi: main_sapi.cpp dwave/external/base64/base64.cpp
	$(CXX) $^ $(CXXFLAGS) -o $(EXE_SAPI) $(LIB_PATH) $(LIBS_SAPI)

qubo: main_qubo.cpp
	$(CXX) $^ $(CXXFLAGS) -o $(EXE_QUBO) $(LIB_PATH)

clean:
	rm -f $(EXE_SAPI)
	rm -f $(EXE_QUBO)

.PHONY: all clean

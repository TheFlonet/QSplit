CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wpedantic -Werror -Weffc++# -Wextra
LIB_PATH = -I./qsplit/external -I./dwave/external

EXE_SAPI = build/sapi
LIBS_SAPI = -lcurl -pthread

EXE_QUBO = build/qubo

FILES = main_qubo.cpp qsplit/qsplit_sampler.hpp qsplit/qubo.hpp qsplit/la_util.hpp main_sapi.cpp dwave/dwave.hpp

all: sapi qubo

sapi: main_sapi.cpp dwave/external/base64/base64.cpp
	$(CXX) $^ $(CXXFLAGS) -o $(EXE_SAPI) $(LIB_PATH) $(LIBS_SAPI)

qubo: main_qubo.cpp
	$(CXX) $^ $(CXXFLAGS) -o $(EXE_QUBO) $(LIB_PATH)

clean:
	rm -f $(EXE_SAPI)
	rm -f $(EXE_QUBO)

check:
	cppcheck -j 8 --check-level=exhaustive --suppressions-list=suppress_check.txt $(FILES)

lint:
	cpplint --linelength=120 $(FILES)

.PHONY: all clean check lint

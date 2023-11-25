#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

using namespace std;

constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

static uint32_t FLOAT_MASK = (1 << 24) - 1;
static float FLOAT_DIVISOR = 1.0f / (1 << 24);

struct mt19937_data_pod {
  uint64_t seed_;
  int left_;
  bool seeded_;
  uint32_t next_;
  std::array<uint32_t, MERSENNE_STATE_N> state_;
};

class mt19937_engine {
public:

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  inline explicit mt19937_engine(uint64_t seed = 5489) {
    init_with_uint32(seed);
  }

  inline mt19937_data_pod data() const {
    return data_;
  }

  inline void set_data(const mt19937_data_pod& data) {
    data_ = data;
  }

  inline uint64_t seed() const {
    return data_.seed_;
  }

  inline bool is_valid() {
    if ((data_.seeded_ == true)
      && (data_.left_ > 0 && data_.left_ <= MERSENNE_STATE_N)
      && (data_.next_ <= MERSENNE_STATE_N)) {
      return true;
    }
    return false;
  }

  inline uint32_t operator()() {
    if (--(data_.left_) == 0) {
        next_state();
    }
    uint32_t y = *(data_.state_.data() + data_.next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
  }

private:
  mt19937_data_pod data_;

  inline void init_with_uint32(uint64_t seed) {
    data_.seed_ = seed;
    data_.seeded_ = true;
    data_.state_[0] = seed & 0xffffffff;
    for (int j = 1; j < MERSENNE_STATE_N; j++) {
      data_.state_[j] = (1812433253 * (data_.state_[j-1] ^ (data_.state_[j-1] >> 30)) + j);
    }
    data_.left_ = 1;
    data_.next_ = 0;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  inline void next_state() {
    uint32_t* p = data_.state_.data();
    data_.left_ = MERSENNE_STATE_N;
    data_.next_ = 0;

    for(int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for(int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
  }

};

int main() {
    // Manually setting the seed
    unsigned int seed = 2147483647;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Generate and print 5 random numbers
    cout << "Built-in generator: ";
    for (int i = 0; i < 5; ++i) {
        double random_number = distribution(generator);
        std::cout << random_number << ", ";
    }
    std::cout << std::endl;

    cout << "Torch generator: ";
    mt19937_engine torch_generator = mt19937_engine(2147483647);
    for (int i = 0; i < 5; ++i) {
        int x = torch_generator();
        float random_float = (x & FLOAT_MASK) * FLOAT_DIVISOR;
        std::cout << random_float << ", ";
    }
    std::cout << std::endl;

    return 0;
}
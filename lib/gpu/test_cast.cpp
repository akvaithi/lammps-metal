#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>

float cast_double_to_float(double d) {
    uint32_t dx[2];
    std::memcpy(dx, &d, 8);
    uint32_t x = dx[0];
    uint32_t y = dx[1];
    
    uint32_t sign = ((y >> 31) & 1) << 31;
    uint32_t exp = (((((y >> 20) & 0x7FF) - 1023) + 127) << 23);
    uint32_t frac = ((y & 0xFFFFF) << 3) | (x >> 29);
    
    uint32_t f_bits = sign | exp | frac;
    float f;
    std::memcpy(&f, &f_bits, 4);
    return f;
}

int main() {
    double vals[] = { 1.6795962, 0.0, -1.0, 15.0, 3.14159 };
    for (double v : vals) {
        float f = cast_double_to_float(v);
        std::cout << v << " -> " << f << std::endl;
    }
    return 0;
}

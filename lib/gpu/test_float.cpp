#include <iostream>
#include <cstdint>
#include <cstring>

int main() {
    double d = 1.6795962; // some typical coordinate
    uint32_t dx[2];
    std::memcpy(dx, &d, 8);
    
    // dx[0] is dx.x, dx[1] is dx.y
    uint32_t sign = (dx[1] >> 31) & 1;
    int32_t exp = (int32_t)((dx[1] >> 20) & 0x7FF) - 1023;
    uint32_t mantissa = ((dx[1] & 0xFFFFF) << 3) | (dx[0] >> 29);
    
    uint32_t f_bits;
    if (exp == -1023) {
        f_bits = 0; // zero or denormal
    } else if (exp <= -127) {
        f_bits = 0; // underflow
    } else {
        f_bits = (sign << 31) | ((exp + 127) << 23) | mantissa;
    }
    
    float f;
    std::memcpy(&f, &f_bits, 4);
    std::cout << "Original double: " << d << "\n";
    std::cout << "Converted float: " << f << "\n";
    
    // Test what happens in my Metal logic exactly:
    uint32_t dxy = dx[1];
    uint32_t dxx = dx[0];
    uint32_t metal_bits = (((dxy >> 31) & 1) << 31) | (((((dxy >> 20) & 0x7FF) - 1023) + 127) << 23) | ((dxy & 0xFFFFF) << 3 | (dxx >> 29));
    float metal_f;
    std::memcpy(&metal_f, &metal_bits, 4);
    std::cout << "Metal formula float: " << metal_f << "\n";
    return 0;
}

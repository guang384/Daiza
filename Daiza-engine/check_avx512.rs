fn main() {
    println!("avx: {}", std::is_x86_feature_detected!("avx"));
    println!("avx2: {}", std::is_x86_feature_detected!("avx2"));
    println!("fma: {}", std::is_x86_feature_detected!("fma"));
    println!("f16c: {}", std::is_x86_feature_detected!("f16c"));
    println!("bmi1: {}", std::is_x86_feature_detected!("bmi1"));
    println!("bmi2: {}", std::is_x86_feature_detected!("bmi2"));
    println!("avx512f: {}", std::is_x86_feature_detected!("avx512f"));
    println!("avx512bw: {}", std::is_x86_feature_detected!("avx512bw"));
    println!("avx512vl: {}", std::is_x86_feature_detected!("avx512vl"));
    println!("avx512vnni: {}", std::is_x86_feature_detected!("avx512vnni"));
    println!("avx512bf16: {}", std::is_x86_feature_detected!("avx512bf16"));
    println!("gfni: {}", std::is_x86_feature_detected!("gfni"));
    println!("vaes: {}", std::is_x86_feature_detected!("vaes"));
    println!("pclmulqdq: {}", std::is_x86_feature_detected!("pclmulqdq"));
}

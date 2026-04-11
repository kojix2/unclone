#[allow(dead_code)]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max_exp = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if max_exp.is_infinite() {
        return max_exp;
    }

    let total: f64 = values.iter().map(|value| (value - max_exp).exp()).sum();
    total.ln() + max_exp
}

#[cfg(test)]
mod tests {
    use super::log_sum_exp;

    #[test]
    fn returns_negative_infinity_for_all_negative_infinity() {
        let actual = log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert!(actual.is_infinite());
        assert!(actual.is_sign_negative());
    }
}

import numpy as np
from full_binary_domain import FullBinaryDomain
from generator import generate_alternate_balanced_binary_vector_one_zero, generate_alternate_balanced_binary_vector_zero_one, generate_half_ones_half_zeros_binary_vector, generate_half_zeros_half_ones_binary_vector, generate_quarter_ones_half_zeros_quarter_ones_binary_vector, generate_quarter_zeros_half_ones_quarter_zeros_binary_vector, generate_quarter_ones_quarter_zeros_half_ones_binary_vector, generate_quarter_zeros_quarter_ones_half_zeros_binary_vector, generate_eighths_alternating_binary_vector, generate_eighths_alternating_binary_vector_starting_with_zero


class WalshTransform:
    def __init__(self,
                 n_bits: int
                 ) -> None:
        self.__domain: FullBinaryDomain = FullBinaryDomain(n_bits)
        self.__number_of_ones_for_each_number: np.ndarray = np.array([bin(i)[2:].count('1') for i in range(self.__domain.space_cardinality())])
        self.__boolean_mask_number_of_ones_for_each_number: dict[int, np.ndarray] = {t: self.__number_of_ones_for_each_number <= t for t in range(self.__domain.number_of_bits() + 1)}
        self.__compute_non_linearity_on_fixed_known_truth_tables()

    def __compute_non_linearity_on_fixed_known_truth_tables(self) -> None:
        self.__fixed_known_truth_tables = []

        tt1 = generate_alternate_balanced_binary_vector_one_zero(self.__domain.space_cardinality())
        tt2 = generate_alternate_balanced_binary_vector_zero_one(self.__domain.space_cardinality())
        tt3 = generate_half_ones_half_zeros_binary_vector(self.__domain.space_cardinality())
        tt4 = generate_half_zeros_half_ones_binary_vector(self.__domain.space_cardinality())
        tt5 = generate_quarter_ones_half_zeros_quarter_ones_binary_vector(self.__domain.space_cardinality())
        tt6 = generate_quarter_zeros_half_ones_quarter_zeros_binary_vector(self.__domain.space_cardinality())
        tt7 = generate_quarter_ones_quarter_zeros_half_ones_binary_vector(self.__domain.space_cardinality())
        tt8 = generate_quarter_zeros_quarter_ones_half_zeros_binary_vector(self.__domain.space_cardinality())
        tt9 = generate_eighths_alternating_binary_vector(self.__domain.space_cardinality())
        tt10 = generate_eighths_alternating_binary_vector_starting_with_zero(self.__domain.space_cardinality())

        for i, tt in enumerate([tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10]):
            spectrum, _ = self.apply(tt)
            nl = self.granular_non_linearity(spectrum)
            self.__fixed_known_truth_tables.append((tt, spectrum, nl))

        self.__best_fixed_known_truth_table = max(self.__fixed_known_truth_tables, key=lambda x: x[2])[0]

    def best_fixed_known_truth_table(self) -> np.ndarray:
        return self.__best_fixed_known_truth_table

    def domain(self) -> FullBinaryDomain:
        return self.__domain

    def number_of_ones_for_each_number(self) -> np.ndarray:
        return self.__number_of_ones_for_each_number

    def boolean_mask_number_of_ones_for_each_number(self, t: int) -> np.ndarray:
        return self.__boolean_mask_number_of_ones_for_each_number[t]

    def resiliency(self, spectrum: np.ndarray) -> int:
        max_resiliency_found_so_far: int = -1
        t: int = 0
        if spectrum[0] == 0:
            max_resiliency_found_so_far = 0
            for iii in range(self.__domain.number_of_bits()):
                t += 1
                m: np.ndarray = self.boolean_mask_number_of_ones_for_each_number(t)
                if np.any(spectrum[m]):
                    return max_resiliency_found_so_far
                else:
                    max_resiliency_found_so_far += 1
                    if iii == self.__domain.number_of_bits() - 1:
                        return max_resiliency_found_so_far
        else:
            return max_resiliency_found_so_far

    def correlation_immunity(self, spectrum: np.ndarray, tol: float = 1e-9) -> int:
        size = 1 << self.__domain.number_of_bits()
        assert spectrum.shape[0] == size, "Walsh spectrum length mismatch"

        max_order = 0
        # For each possible order
        for m in range(1, self.__domain.number_of_bits() + 1):
            ok = True
            for idx in range(1, size):  # exclude index 0
                if bin(idx).count("1") <= m and abs(spectrum[idx]) > tol:
                    ok = False
                    break
            if ok:
                max_order = m
            else:
                break
        return max_order

    def non_linearity(self, spectrum: np.ndarray) -> float:
        x: np.ndarray = np.absolute(spectrum)
        m: float = float(np.max(x))
        return  self.__domain.space_cardinality() / 2.0 - 0.5 * m

    def granular_non_linearity(self, spectrum: np.ndarray) -> float:
        x: np.ndarray = np.absolute(spectrum)
        m: float = float(np.max(x))
        max_values: int = int((x == m).sum())
        nl: float = self.__domain.space_cardinality() / 2.0 - 0.5 * m
        closeness: float = (self.__domain.space_cardinality() -  max_values) / self.__domain.space_cardinality()
        return nl + closeness

    def apply(self, result: np.ndarray) -> tuple[np.ndarray, int]:
        return self.__fast_walsh_transform_init(result)

    def invert(self, spectrum: np.ndarray, directly_go_to_truth_table: bool = False) -> tuple[np.ndarray, int]:
        return self.__inverse_fast_walsh_transform_init(spectrum, directly_go_to_truth_table)

    def __fast_walsh_transform_init(self, result: np.ndarray) -> tuple[np.ndarray, int]:
        polar_form: np.ndarray = FullBinaryDomain.convert_truth_table_to_polar_form(result)
        l: list[int] = polar_form.tolist()
        spectral_radius: int = self.__fast_walsh_transform(l, 0, len(l))
        return np.array(l), spectral_radius

    def __fast_walsh_transform(self, v: list[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] += v[i + half]
            v[i + half] = temp - v[i + half]

        if half > 1:
            val1: int = self.__fast_walsh_transform(v, start, half)
            val2: int = self.__fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if abs(v[start]) > abs(v[start + half]):
                return abs(v[start])
            else:
                return abs(v[start + half])

    def __inverse_fast_walsh_transform_init(self, result: np.ndarray, directly_go_to_truth_table: bool = False) -> tuple[np.ndarray, int]:
        l: list[int] = result.tolist()
        max_auto_correlation_coefficient: int = self.__inverse_fast_walsh_transform(l, 0, len(l))
        r: np.ndarray = np.array(l, dtype=np.int64)
        if not directly_go_to_truth_table:
            return r, max_auto_correlation_coefficient
        return FullBinaryDomain.convert_polar_form_to_truth_table(r), max_auto_correlation_coefficient

    def __inverse_fast_walsh_transform(self, v: list[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] = int( (v[i] + v[i + half]) / 2.0 )
            v[i + half] = int( (temp - v[i + half]) / 2.0 )

        if half > 1:
            val1: int = self.__inverse_fast_walsh_transform(v, start, half)
            val2: int = self.__inverse_fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if start == 0:
                return abs(v[1])
            else:
                if abs(v[start]) > abs(v[start + half]):
                    return abs(v[start])
                else:
                    return abs(v[start + half])


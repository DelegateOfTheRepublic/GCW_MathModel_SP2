from enum import StrEnum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from outliers import smirnov_grubbs as grubbs


class Model:
    __data: dict = None
    __factors: list[str] = None
    __predicted_y: np.ndarray = None
    __corr_matrix_values: np.ndarray = None
    __regression_line_coeffs: tuple = None
    __dep_variable: str = None

    class __CouplingStrength(StrEnum):
        VERY_LOW = 'Очень слабая'
        LOW = 'Слабая'
        MIDDLE = 'Средняя'
        HIGH = 'Высокая'
        VERY_HIGH = 'Очень высокая'

    def __init__(self):
        pd_data = pd.read_excel('data.xlsx').to_dict()

        for key, value in pd_data.items():
            pd_data[key] = np.array(list(pd_data[key].values()))

        self.__data = pd_data

        data_keys = list(self.__data.keys())
        self.__dep_variable = data_keys.pop()
        self.__factors = data_keys

    @property
    def data(self):
        return self.__data

    @property
    def factors(self) -> list[str]:
        return self.__factors

    @property
    def predicted_y(self) -> np.ndarray:
        return self.__predicted_y

    @property
    def dep_variable(self) -> str:
        return self.__dep_variable

    @property
    def n_sample(self) -> int:
        return len(self.__data[self.__dep_variable])

    @property
    def dep_variable_values(self) -> np.ndarray:
        return self.__data[self.__dep_variable]

    def anomaly_check(self):
        anomaly_objects: list[int] = []

        for factor_key in self.__factors:
            anomaly_value: list = grubbs.max_test_outliers(self.__data[factor_key], alpha=.05)
            anomaly_object: list = grubbs.max_test_indices(self.__data[factor_key], alpha=.05)
            if anomaly_value:
                anomaly_objects += anomaly_object

                print(f'[Внимание!]Выброс в данных фактора {factor_key}\n\t[Номер объекта] {anomaly_object[0]}'
                      f'\n\t[Значение объекта] {anomaly_value[0]}'
                      f'\n\t[Значения объекта в других факторах] {' '.join([str(self.__data[_][anomaly_object[0]])
                                                                            for _ in self.__factors
                                                                            if _ != factor_key])}\n')

        if len(anomaly_objects) == 0:
            print('[Внимание!]Аномальные объекты не обнаружены.')
        else:
            print(f'[Внимание!]Аномальные объекты будут удалены автоматически.')
            self.__delete_anomaly_objects(anomaly_objects)

    def corr_matrix_check(self):
        multicol_pairs: list[list[tuple]] = []
        factors_dep_var: list[str] = []
        corr_matrix: np.ndarray = self.__corr_matrix()

        for i in range(len(self.__factors) - 1):
            for j in range(i + 1, len(self.__factors)):
                coupling_strength = self.__coupling_strength_check(abs(corr_matrix[i, j]))
                if coupling_strength['is_multicol']:
                    multicol_pairs.append([(self.__factors[i], corr_matrix[i, -1]),
                                           (self.__factors[j], corr_matrix[j, -1])])
                    print(
                        f'[Внимание!]Мультиколлинеарность между факторами {self.__factors[i]} и {self.__factors[j]}.')
                    print(f'\t[Значение коэффициента корреляции по модулю] {abs(corr_matrix[i, j])}')

        if len(multicol_pairs) == 0:
            print('[Внимание!]Мультиколлинеарности не обнаружено.')
        else:
            print(f'[Внимание!]Мультиколлинеарность будет устранена автоматически.')

            for pair in multicol_pairs:
                factor_a, factor_a_value = pair[0]
                factor_b, factor_b_value = pair[1]

                if abs(factor_a_value) > abs(factor_b_value):
                    self.__delete_factors([factor_b])
                else:
                    self.__delete_factors([factor_a])

        for i, factor in enumerate(self.__factors):
            coupling_strength = self.__coupling_strength_check(abs(corr_matrix[i, -1]))
            if coupling_strength['coupling_strength'] in [self.__CouplingStrength.VERY_LOW,
                                                          self.__CouplingStrength.LOW]:
                print(f'[Внимание!]{coupling_strength['coupling_strength']} связь между фактором {factor} '
                      f'и зависимой переменной {self.__dep_variable}')
                factors_dep_var.append(factor)

        if len(factors_dep_var) == 0:
            print(f'[Внимание!]Факторы со слабой связью с зависимой переменной не обнаружены.')
        else:
            print(f'[Внимание!]Факторы со слабой связью с зависимой переменной будут удалены автоматически.')

            self.__delete_factors(factors_dep_var)

        self.__corr_matrix_values = self.__corr_matrix()

    def plot_dep_var_factors(self):
        plt.figure(figsize=(18, 5))

        for i, factor in enumerate(self.__factors, 1):
            plt.subplot(1, len(self.__factors), i)
            plt.plot(self.__data[factor], self.dep_variable_values, marker='o', color='blue')
            plt.title(f'Linear Regression for {factor}')
            plt.xlabel(factor)
            plt.ylabel(self.dep_variable)

        plt.tight_layout()
        plt.show()

    def plot_trendline(self, factor_x: str, factor_y: str):
        plt.figure(figsize=(8, 8))
        new_y: np.ndarray = self.__regression_line(self.__data[factor_x], self.__data[factor_y])
        plt.scatter(self.__data[factor_x], self.__data[factor_y])
        plt.xlabel(factor_x)
        plt.ylabel(factor_y)
        plt.plot(self.__data[factor_x], new_y, color='g', linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_corr_matrix(self):
        pass

    def regression_stat(self):
        if self.__regression_line_coeffs is None:
            self.__multi_regression_line()

        print(f'[Множественный R] {self.__r_coeff(self.__predicted_y)}\n'
              f'[R-квадрат] {self.__r_square(self.__predicted_y)}\n'
              f'[Нормированный R-квадрат] {self.__adjusted_r(self.__predicted_y)}\n'
              f'[Стандартная ошибка] {self.__standard_error(self.__predicted_y)}\n'
              f'[Размер выборки] {self.n_sample}')

    def regression_analysis(self):
        new_data: np.ndarray = np.ones((len(self.__factors) + 1, self.n_sample), dtype=float)

        for i, factor in enumerate(self.__factors):
            new_data[i] = self.__data[factor]

        if self.__predicted_y is None:
            self.__multi_regression_line()

        variance_errors: float = np.sqrt(np.sum((self.dep_variable_values - self.__predicted_y) ** 2) /
                                         (self.n_sample - len(self.__factors) - 1))

        standart_factors_errors: np.ndarray = variance_errors * np.sqrt(np.linalg.inv(new_data @ new_data.transpose())
                                                                        .diagonal())

        t_stats: np.ndarray = np.abs(self.__regression_line_coeffs) / standart_factors_errors
        p_values: np.ndarray = (1 - stats.t.cdf(np.abs(t_stats), self.n_sample - len(self.__factors) - 1)) * 2
        margin_errors: np.ndarray = stats.t.isf(0.05 / 2, 20)
        confidence_interval: dict[str, np.ndarray] = {
            'left': self.__regression_line_coeffs - margin_errors * standart_factors_errors,
            'right': self.__regression_line_coeffs + margin_errors * standart_factors_errors
        }
        print(self.__regression_line_coeffs)
        print(standart_factors_errors)
        print(t_stats)
        print(p_values)
        print(confidence_interval['left'], confidence_interval['right'])

    def regression_equation(self) -> str:
        to_print: str = f'~y = {self.__regression_line_coeffs[-1]:.4f}'

        for i, factor in enumerate(self.__factors):
            sign: str = '+'
            if self.__regression_line_coeffs[i] < 0:
                sign = '-'

            to_print += f' {sign} {abs(self.__regression_line_coeffs[i]):.4f}*{factor}'

        return to_print

    def __adjusted_r(self, predicted_y: np.ndarray) -> float:
        return 1 - (self.__standard_error(predicted_y)) / \
            (np.sum((self.dep_variable_values - self.dep_variable_values.mean()) ** 2) / (self.n_sample - 1))

    def __corr_matrix(self, data: np.ndarray = None) -> np.ndarray:
        new_data: np.ndarray = data or np.array(list(self.__data.values()))
        n: int = len(new_data)
        corr_matrix: np.ndarray = np.eye(n, n, dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                corr_matrix[i][j] = self.__correlation(new_data[i], new_data[j])
                corr_matrix[j][i] = corr_matrix[i][j]

        return corr_matrix

    def __coupling_strength_check(self, corr_coef: float) -> dict:
        if corr_coef <= 0.2: return {'coupling_strength': self.__CouplingStrength.VERY_LOW, 'is_multicol': False}
        if .2 < corr_coef <= .5: return {'coupling_strength': self.__CouplingStrength.LOW, 'is_multicol': False}
        if .5 < corr_coef <= .7: return {'coupling_strength': self.__CouplingStrength.MIDDLE, 'is_multicol': False}
        if .7 < corr_coef <= .8: return {'coupling_strength': self.__CouplingStrength.HIGH, 'is_multicol': False}

        return {'coupling_strength': self.__CouplingStrength.VERY_HIGH, 'is_multicol': True}

    def __covariance(self, x: np.ndarray, y: np.ndarray) -> float:
        dx: np.ndarray = x - x.mean()
        dy: np.ndarray = y - y.mean()

        return (dx * dy).sum() / (len(dx) - 1)

    def __correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.__covariance(x, y) / (self.__standard_deviation(x) * self.__standard_deviation(y))

    def __delete_anomaly_objects(self, anomaly_objects: list):
        for factor_key in self.__factors:
            self.__data[factor_key] = np.delete(self.__data[factor_key], anomaly_objects)

        self.__data[self.__dep_variable] = np.delete(self.dep_variable_values, anomaly_objects)
        print(f'\t[Внимание!]Аномальные объекты '
              f'{', '.join([str(anomaly_object + 1) for anomaly_object in anomaly_objects])} удалены.')

    def __delete_factors(self, factors: list[str]):
        for factor in factors:
            self.__data.pop(factor)
            print(f'\t[Внимание!]Фактор {factor} удален!')

        self.__update_factors()

    def __multi_regression_line(self):
        new_data: np.ndarray = np.ones((len(self.__factors) + 1, self.n_sample), dtype=float)
        for i, factor in enumerate(self.__factors):
            new_data[i] = self.__data[factor]

        transpose_new_data: np.ndarray = new_data.transpose()

        regression_line: np.ndarray = (np.linalg.inv(new_data @ transpose_new_data)
                                       @ new_data @ self.dep_variable_values[np.newaxis].T).T[0]

        self.__predicted_y = np.zeros(self.n_sample, dtype=float)
        for i, factor in enumerate(self.__factors):
            self.__predicted_y += regression_line[i] * self.__data[factor]

        self.__predicted_y += regression_line[-1]
        self.__regression_line_coeffs = tuple(regression_line)

    def predicted_y_errors(self):
        errors: np.ndarray = np.abs(self.dep_variable_values - self.__predicted_y) / self.dep_variable_values * 100
        print(', '.join([f'{error:.1f}%' for error in errors]))

    def __r_coeff(self, predicted_y: np.ndarray) -> float:
        return (np.sum((self.dep_variable_values - self.dep_variable_values.mean()) *
                       (predicted_y - predicted_y.mean())) /
                np.sqrt(np.sum((self.dep_variable_values - self.dep_variable_values.mean()) ** 2) *
                        np.sum((predicted_y - predicted_y.mean()) ** 2)))

    def __r_square(self, predicted_y: np.ndarray) -> float:
        return self.__r_coeff(predicted_y) ** 2

    def __regression_line(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n: int = np.size(x)

        m_x: float = np.mean(x)
        m_y: float = np.mean(y)

        ss_xy: float = np.sum(y * x) - n * m_y * m_x
        ss_xx: float = np.sum(x * x) - n * m_x * m_x

        b_1: float = ss_xy / ss_xx
        b_0: float = m_y - b_1 * m_x

        return b_0 + b_1 * x

    def __standard_error(self, x: np.ndarray) -> float:
        return np.sqrt(np.sum((self.dep_variable_values - x) ** 2) / (self.n_sample - len(self.__factors) - 1))

    def __standard_deviation(self, x: np.ndarray) -> float:
        return np.sqrt(self.__variance(x))

    def __update_factors(self):
        head: list[str] = list(self.__data.keys())
        head.pop()
        self.__factors = head

    def __variance(self, x: np.ndarray) -> float:
        x_hat: float = x.mean()
        n = len(x)
        n = n - 1 if n in range(1, 30) else n

        return sum((x - x_hat) ** 2) / n

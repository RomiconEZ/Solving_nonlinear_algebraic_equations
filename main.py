import numpy as np
import math
from datetime import datetime
import time
import sympy
np.seterr(all='ignore')

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
        L[i, i] = 1
        L[i, i + 1:] = 0
    return np.matrix(L)


def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return np.matrix(U)


def gauss(x):
    n = len(x)
    if abs(x[0]) > 1e-8:
        t = x[1:n] / x[0]  # Находим коэффициенты по столбцу для обнуления стобца под элементом
    else:
        t = np.zeros((n - 1,))
    return t


def gauss_app(C, t, k, raz):
    n = C.shape[0]
    for i in range(1, n):
        C[i, :] = C[i, :] - np.multiply(t[i - 1], C[0, :])  # Вычитаем строку, умноженную на коэф. из всех строк ниже ее
    return C


def perestanovka_strok_stolbcov(A, k, raz, kolvo_nech_perest):
    ind = np.unravel_index(np.argmax(abs(A[k:, k:])),
                           A[k:, k:].shape)  # нахожу индекс максимального элемента в урезанной матрице A
    ind = list(ind)
    ind[0] += k  # прибавляю k, чтобы найти номера строки и столбца в основной матрице A
    ind[1] += k

    Mp = np.eye(raz)
    if ind[0] != 0:  # Если были перестановки строк в матрице
        Mp[k, k] = 0
        Mp[ind[0], ind[0]] = 0
        Mp[k, ind[0]] = 1
        Mp[ind[0], k] = 1
        if kolvo_nech_perest == True:
            kolvo_nech_perest = False
        else:
            kolvo_nech_perest = True

    Mq = np.eye(raz)

    if ind[1] != 0:  # Если были перестановки столбцов в матрице
        Mq[k, k] = 0
        Mq[ind[1], ind[1]] = 0
        Mq[k, ind[1]] = 1
        Mq[ind[1], k] = 1
        if kolvo_nech_perest == True:
            kolvo_nech_perest = False
        else:
            kolvo_nech_perest = True

    return [Mp, Mq, kolvo_nech_perest]


def IsklyucheniyeGaussaSVneshnimProizvedeniyem(A):
    n = A.shape[0]
    Mp = np.eye(n)  # Общая матрица перестановки строк
    Mq = np.eye(n)  # Общая матрица перестановки столбцов
    kolvo_nech_perest = True
    # False - нечетное количество нечетных перестановок
    # True - четное количество нечетных перестановок

    # Подсчитывается количетство нечетных перестановок в матрице, так как для перемещения двух произвольных
    # строк/столбцов в матрице необходимо 2(s-k)-1 транспозиция, соответственно нечетное количество
    # нечетных перестановок даст нам нечетное количество транспозиция в целом, тогда определитель поменяет знак
    arithmetic_oper = 0
    for k in range(0, n):
        M_p_q_per = perestanovka_strok_stolbcov(A, k, n, kolvo_nech_perest)

        kolvo_nech_perest = M_p_q_per[2]

        # Переставляем строки и столбцы в матрице A
        A = np.matmul(M_p_q_per[0], A)
        A = np.matmul(A, M_p_q_per[1])

        # Меняем общиие матрицы перестановок для A
        Mp = np.matmul(M_p_q_per[0], Mp)
        Mq = np.matmul(Mq, M_p_q_per[1])

        t = gauss(A[k:n, k])
        # меняем вектор коэффициентов под рассматриваемым элементом на главной диагонали
        arithmetic_oper += (n - k)
        A[k + 1:n, k] = t
        A[k:n, k + 1:n] = gauss_app(A[k:n, k + 1:n], t, k, n)
        arithmetic_oper = (n - k) + (n - k) * (n - k)
        # вычитаем урезанную строку, умноженную на коэф. из низ лежащих строк

    return [A, Mp, Mq, int(kolvo_nech_perest), arithmetic_oper]


def solve(Check, A, b, LU, P, Q):
    L = get_L(LU)
    U = get_U(LU)
    b_init = b
    b = np.matmul(P, b)
    y = np.empty((b.shape[0], 1))
    x = np.empty((b.shape[0], 1))
    for i in range(0, b.shape[0]):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    if Check[0] == False:
        for i in range(b.shape[0] - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    else:
        # for j in range (0,(A.shape[0]-Check[1])-1):
        #     x[j]=0
        for i in range(b.shape[0] - 1 - (A.shape[0] - Check[1]), -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    x_init = np.matmul(Q, x)
    if not np.isclose(np.matmul(A, x_init), b_init, rtol=1e-05, atol=1e-08, equal_nan=False).all():
        print('Неправильный ответ')
    return x_init


def degenerate_check_and_rang(U):
    kolvo_nul = 0
    for i in range(0, U.shape[0]):
        if abs(U[i, i]) < 1e-10:
            kolvo_nul += 1
    if kolvo_nul != 0:
        return [True, U.shape[0] - kolvo_nul]
    else:
        return [False, U.shape[0]]


def Function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    F = np.array([
        np.cos(x2 * x1) - np.exp(-3 * x3) + x4 * x5 ** 2 - x6 - np.sinh(
            2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
        np.sin(x2 * x1) + x3 * x9 * x7 - np.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * np.cos(-x9 + x4) + x5 / (x3 + x1) - np.sin(x2 ** 2) + np.cos(
            x7 * x10) ** 2 - x8 - 0.1707472705022304757,
        np.sin(x5) + 2 * x8 * (x3 + x1) - np.exp(-x7 * (-x10 + x6)) + 2 * np.cos(x2) - 1.0 / (
                    -x9 + x4) - 0.3685896273101277862,
        np.exp(x1 - x4 - x9) + x5 ** 2 / x8 + np.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
        x2 ** 3 * x7 - np.sin(x10 / x5 + x8) + (x1 - x6) * np.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6) ** 2 - 2 * np.sin(-x9 + x3) + 0.15e1 * x4 - np.exp(
            x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + np.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + np.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096], dtype=np.float64)
    return F


def Function1(x):  # 12 функция в мет. пособии 1
    F = np.array([np.sqrt(1 + x * x) * (np.sin(3 * x + 0.1) + np.cos(2 * x + 0.3))], dtype=np.float64).reshape(1, 1)
    return F


# def derFunction1(x_):#! изменить, чтобы не было пересчета производной
#     x = sympy.symbols('x')
#     dif = sympy.diff(sympy.sqrt(1+x*x)*(sympy.sin(3*x+0.1)+sympy.cos(2*x+0.3)))
#     dif = np.float64((dif).subs(x,x_[0,0]))
#     dif = np.array([dif], dtype=np.float64).reshape(1,1)
#     return dif
def derFunction1(x):
    dif = np.array(
        [np.sqrt(x ** 2 + 1) * (-2 * np.sin(2 * x + 0.3) + 3 * np.cos(3 * x + 0.1)) * x * (np.sin(3 * x + 0.1) +
                                                                                           np.cos(
                                                                                               2 * x + 0.3)) / np.sqrt(
            x ** 2 + 1)], dtype=np.float64).reshape(1, 1)
    return dif


def Jacobi_matrix(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    J = np.array([[-x2 * np.sin(x2 * x1), -x1 * np.sin(x2 * x1), 3 * np.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                   -1, 0, -2 * np.cosh(2 * x8) * x9, -np.sinh(2 * x8), 2],
                  [x2 * np.cos(x2 * x1), x1 * np.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                   -np.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, np.exp(-x10 + x6)],
                  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                  [-x5 / (x3 + x1) ** 2, -2 * x2 * np.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * np.sin(-x9 + x4),
                   1.0 / (x3 + x1), 0, -2 * np.cos(x7 * x10) * x10 * np.sin(x7 * x10), -1,
                   2 * np.sin(-x9 + x4), -2 * np.cos(x7 * x10) * x7 * np.sin(x7 * x10)],
                  [2 * x8, -2 * np.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, np.cos(x5),
                   x7 * np.exp(-x7 * (-x10 + x6)), -(x10 - x6) * np.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                   -1.0 / (-x9 + x4) ** 2, -x7 * np.exp(-x7 * (-x10 + x6))],
                  [np.exp(x1 - x4 - x9), -1.5 * x10 * np.sin(3 * x10 * x2), -x6, -np.exp(x1 - x4 - x9),
                   2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -np.exp(x1 - x4 - x9),
                   -1.5 * x2 * np.sin(3 * x10 * x2)],
                  [np.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * np.sin(x4),
                   x10 / x5 ** 2 * np.cos(x10 / x5 + x8),
                   -np.cos(x4), x2 ** 3, -np.cos(x10 / x5 + x8), 0, -1.0 / x5 * np.cos(x10 / x5 + x8)],
                  [2 * x5 * (x1 - 2 * x6), -x7 * np.exp(x2 * x7 + x10), -2 * np.cos(-x9 + x3), 1.5,
                   (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * np.exp(x2 * x7 + x10), 0,
                   2 * np.cos(-x9 + x3),
                   -np.exp(x2 * x7 + x10)],
                  [-3, -2 * x8 * x10 * x7, 0, np.exp(x5 + x4), np.exp(x5 + x4),
                   -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
                  [x10, x9, -x8, np.cos(x4 + x5 + x6) * x7, np.cos(x4 + x5 + x6) * x7,
                   np.cos(x4 + x5 + x6) * x7, np.sin(x4 + x5 + x6), -x3, x2, x1]], dtype=np.float64)
    return J


# x_(k+1)=x_(k) - J^(-1) ( x_(k) ) * F( x_(k) )
def Newton_Method(Function, Jacobi_matrix, start, acc):
    try:
        arithmetic_oper = 0
        iter = 1
        xn = start
        Jacobi_mat = Jacobi_matrix(*xn)
        LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(Jacobi_mat.copy())
        arithmetic_oper += arithmetic_oper_LU
        U = get_U(LU)
        Check = degenerate_check_and_rang(U)
        xn1 = xn - solve(Check, Jacobi_mat, Function(*xn), LU, P, Q)
        arithmetic_oper += Jacobi_mat.shape[0] + Jacobi_mat.shape[0] * Jacobi_mat.shape[1] + Jacobi_mat.shape[0] * (
                    Jacobi_mat.shape[1] - 1)
        while (max(abs(xn1 - xn)) > acc) & (iter < 1000):
            xn = xn1
            Jacobi_mat = Jacobi_matrix(*xn)
            LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
                Jacobi_mat.copy())
            arithmetic_oper += arithmetic_oper_LU
            U = get_U(LU)
            Check = degenerate_check_and_rang(U)
            xn1 = xn - solve(Check, Jacobi_mat, Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_mat.shape[0] + Jacobi_mat.shape[0] * Jacobi_mat.shape[1] + Jacobi_mat.shape[0] * (
                        Jacobi_mat.shape[1] - 1)
            iter += 1
        return [xn1, iter, arithmetic_oper]
    except ValueError:
        print(f"Value not invalidate")


def Modify_Newton_Method(Function, Jacobi_matrix, start, acc):
    try:
        arithmetic_oper = 0
        iter = 1
        xn = start
        Jacobi_matrix_X_O = Jacobi_matrix(*start)
        LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
            Jacobi_matrix_X_O.copy())
        arithmetic_oper += arithmetic_oper_LU
        U = get_U(LU)
        Check = degenerate_check_and_rang(U)
        if (Check[0] != True):
            delta_x = solve(Check, Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
            # умножение строки на вектор коэффициентов, а потом вычитание ( делим на 2(матрица треугольная), умножаем на 2 (2 операции) )
        else:
            raise Exception("Матрица вырожденная")
        xn1 = xn + delta_x
        while ((max(abs(xn1 - xn)) > acc) | (max(Function(*xn1)) > acc)) & (iter < 1000):
            xn = xn1
            delta_x = solve(Check, Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
            xn1 = xn + delta_x
            iter += 1
        return [xn1, iter, arithmetic_oper]
    except ValueError:
        print(f"Value not invalidate")


def Newton_Method_Hybrid(Function, Jacobi_matrix, start, acc, num_for_chan):
    try:
        arithmetic_oper = 0
        iter = 1
        xn = start
        Jacobi_mat = Jacobi_matrix(*start)
        LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(Jacobi_mat.copy())
        arithmetic_oper += arithmetic_oper_LU
        U = get_U(LU)
        Check = degenerate_check_and_rang(U)
        xn1 = xn - solve(Check, Jacobi_mat, Function(*xn), LU, P, Q)
        arithmetic_oper += Jacobi_mat.shape[0] + Jacobi_mat.shape[0] * Jacobi_mat.shape[1] + Jacobi_mat.shape[0] * (
                    Jacobi_mat.shape[1] - 1)
        while (max(abs(xn1 - xn)) > acc) & (iter < num_for_chan):
            xn = xn1

            Jacobi_mat = Jacobi_matrix(*xn)
            LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
                Jacobi_mat.copy())
            arithmetic_oper += arithmetic_oper_LU
            U = get_U(LU)
            Check = degenerate_check_and_rang(U)
            xn1 = xn - solve(Check, Jacobi_mat, Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_mat.shape[0] + Jacobi_mat.shape[0] * Jacobi_mat.shape[1] + Jacobi_mat.shape[0] * (
                    Jacobi_mat.shape[1] - 1)

            iter += 1
        xn = xn1
        Jacobi_matrix_X_O = Jacobi_matrix(*xn)
        LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
            Jacobi_matrix_X_O.copy())
        arithmetic_oper += arithmetic_oper_LU
        delta_x = solve([False, LU.shape[0]], Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
        arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
        xn1 = xn + delta_x
        while ((max(abs(xn1 - xn)) > acc) | (max(Function(*xn1)) > acc)) & (iter < 1000):
            xn = xn1
            delta_x = solve([False, LU.shape[0]], Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
            xn1 = xn + delta_x
            iter += 1
        return [xn1, iter, arithmetic_oper]
    except ValueError:
        print(f"Value not invalidate")


def Modify_Newton_Method_Recount_Jac_every_k_iter(Function, Jacobi_matrix, start, acc, k):
    try:
        arithmetic_oper = 0
        iter = 1
        xn = start
        Jacobi_matrix_X_O = Jacobi_matrix(*start)
        LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
            Jacobi_matrix_X_O.copy())
        arithmetic_oper += arithmetic_oper_LU
        U = get_U(LU)
        Check = degenerate_check_and_rang(U)
        if (Check[0] != True):
            delta_x = solve(Check, Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
        else:
            raise Exception("Матрица вырожденная")
        xn1 = xn + delta_x
        while ((max(abs(xn1 - xn)) > acc) | (max(Function(*xn1)) > acc)) & (iter < 1000):
            xn = xn1
            if iter % k == 0:
                Jacobi_matrix_X_O = Jacobi_matrix(*xn)
                LU, P, Q, kolvo_nech_perest, arithmetic_oper_LU = IsklyucheniyeGaussaSVneshnimProizvedeniyem(
                    Jacobi_matrix_X_O.copy())
                arithmetic_oper += arithmetic_oper_LU
            delta_x = solve(Check, Jacobi_matrix_X_O, -Function(*xn), LU, P, Q)
            arithmetic_oper += Jacobi_matrix_X_O.shape[0] * Jacobi_matrix_X_O.shape[1]
            xn1 = xn + delta_x
            iter += 1
        return [xn1, iter, arithmetic_oper]
    except ValueError:
        print(f"Value not invalidate")


print(f'----------------------------------------------------')
print(f'Задание 1: алгоритм Ньютона для нелинейного алгебраического уравнения')
start_time = datetime.now()
start1 = np.array([[0.]], dtype=np.float64)
ans = Newton_Method(Function1, derFunction1, start1, 1e-10)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Проверка F*x=')
print(f'{Function1(*ans[0])}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')

print(f'----------------------------------------------------')

print(f'Задание 2: алгоритм Ньютона для системы нелинейных алгебраических уравнений')
start = np.array([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]], dtype=np.float64)
start_time = datetime.now()
ans = Newton_Method(Function, Jacobi_matrix, start, 1e-10)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Проверка F*x=')
print(f'{Function(*ans[0])}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')

print(f'----------------------------------------------------')

print(f'Задание 3: модифицированный алгоритм Ньютона')
start_time = datetime.now()
ans = Modify_Newton_Method(Function, Jacobi_matrix, start, 1e-10)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Проверка F*x=')
print(f'{Function(*ans[0])}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')

print(f'----------------------------------------------------')

print(f'Задание 4: гибридный алгоритм Ньютона с переходом на модифицированный алгоритм')
start_time = datetime.now()
num_for_chan = 4
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-10, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Проверка F*x=')
print(f'{Function(*ans[0])}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')

print(f'----------------------------------------------------')

print(f'Задание 5: модифицированный алгоритм Ньютона c пересчитыванием матрицы Якоби каждые 10 итераций')
start_time = datetime.now()
k = 5
ans = Modify_Newton_Method_Recount_Jac_every_k_iter(Function, Jacobi_matrix, start, 1e-10, k)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Проверка F*x=')
print(f'{Function(*ans[0])}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')

print(f'----------------------------------------------------')

print(f'Задание 6: изменение начальных условий')
start = np.array([[0.5], [0.5], [1.5], [-1], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]], dtype=np.float64)
start_time = datetime.now()
num_for_chan = 3
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-8, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Кол-во итераций для перехода: {num_for_chan}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')
print(f'----------------------------------------------------')
start_time = datetime.now()
num_for_chan = 5
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-8, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Кол-во итераций для перехода: {num_for_chan}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')
print(f'----------------------------------------------------')
start_time = datetime.now()
num_for_chan = 7
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-8, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Кол-во итераций для перехода: {num_for_chan}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')
print(f'----------------------------------------------------')
start_time = datetime.now()
num_for_chan = 8
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-8, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Кол-во итераций для перехода: {num_for_chan}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')
print(f'----------------------------------------------------')
start_time = datetime.now()
num_for_chan = 10
ans = Newton_Method_Hybrid(Function, Jacobi_matrix, start, 1e-8, num_for_chan)
total_time = datetime.now() - start_time
print(f'Ответ:')
print(f'{ans[0]}')
print(f'Кол-во итераций для перехода: {num_for_chan}')
print(f'Общее время решения: {total_time}')
print(f'Количество итераций: {ans[1]}')
print(f'Количество арифметических операций: {ans[2]}')
print(f'----------------------------------------------------')

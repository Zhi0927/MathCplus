#ifndef ZHI_MAT_H_
#define ZHI_MAT_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>

#define EPS 1e-10

#ifndef NDEBUG
#define ASSERT(condition, message)                                                                          \
    do {                                                                                                    \
        if (! (condition)) {                                                                                \
            std::cerr << "Assertion failed`" #condition "` at " << __FILE__                                 \
                      << "function " << __FUNCTION__ <<": " << __LINE__ << "."<<std::endl                   \
                      << message << std::endl;                                                              \
            std::terminate();                                                                               \
        }                                                                                                   \
    } while (false)
#else
#define ASSERT(condition, message) do {} while (false)
#endif


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct Matrix {
public:

    Matrix()
        : m_rows(1),
          m_cols(1),
          m_size(1)
    {
        m_data = new T * [m_rows];
        for (size_t i = 0; i < m_rows; ++i) {
            m_data[i] = new T[m_cols];
        }

        m_data[0][0] = 0;
    }

    Matrix(const size_t rows, const size_t cols) 
        : m_rows(rows),
          m_cols(cols),
          m_size(rows * cols)
    {
        m_data = new T * [m_rows];
        for (size_t i = 0; i < m_rows; ++i) {
            m_data[i] = new T[m_cols];
        }

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] = 0;
            }
        }
    }

    Matrix(const size_t rows, const size_t cols, T** data)
        : m_rows(rows),
          m_cols(cols),
          m_size(rows* cols)
    {
        m_data = new T * [m_rows];
        for (size_t i = 0; i < m_rows; ++i) {
            m_data[i] = new T[m_cols];
        }

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] = data[i][j];
            }
        }
    }

    template <size_t rows, size_t cols>
    Matrix(T(&Array)[rows][cols])
        : m_rows(rows),
          m_cols(cols),
          m_size(rows* cols)
    {
        m_data = new T * [m_rows];
        for (size_t i = 0; i < m_rows; ++i) {
            m_data[i] = new T[m_cols];
        }

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] = Array[i][j];
            }
        }
    }

    Matrix(const Matrix& _matrix)
        : m_rows(_matrix.rows()),
          m_cols(_matrix.cols()),
          m_size(_matrix.rows()* _matrix.cols())
    {
        m_data = new T * [m_rows];
        for (size_t i = 0; i < m_rows; ++i) {
            m_data[i] = new T[m_cols];
        }

        for (int i = 0; i < m_rows; ++i) {
            for (int j = 0; j < m_cols; ++j) {
                m_data[i][j] = _matrix.m_data[i][j];
            }
        }
    }

    static Matrix Identity(size_t size)
    {
        Matrix temp(size, size);
        for (size_t i = 0; i < temp.rows(); ++i) {
            for (size_t j = 0; j < temp.cols(); ++j) {
                if (i == j) {
                    temp.m_data[i][j] = 1;
                }
                else {
                    temp.m_data[i][j] = 0;
                }
            }
        }
        return temp;
    }

    ~Matrix() {
        //for (size_t i = 0; i < m_rows; ++i) {
        //    delete[] m_data[i];
        //}
        //delete[] m_data;

        delete[]* m_data, delete[] m_data;
    }


    const size_t rows() const {
        return m_rows;
    }

    const size_t cols() const {
        return m_cols;
    }

    const size_t size() const {
        return m_size;
    }

    Matrix getMatrix(const std::vector<size_t>& rowsIndexArray, const size_t begin_col, const size_t end_col) const {
        Matrix ret(rowsIndexArray.size(), (end_col - begin_col + 1));
        for (size_t i = 0; i < ret.rows(); ++i) {
            for (size_t j = begin_col; j <= end_col; ++j) {
                ret(i, j - begin_col) = m_data[rowsIndexArray[i]][j];
            }
        }
        return ret;
    }

    Matrix getSubMatrix(const size_t begin_row, const size_t end_row, const size_t begin_col, const size_t end_col) const {
        Matrix ret((end_row - begin_row + 1), (end_col - begin_col + 1));

        for (size_t i = begin_row; i <= end_row; ++i) {
            for (size_t j = begin_col; j <= end_col; ++j) {
                ret((i - begin_row), (j - begin_col)) = m_data[i][j];
            }
        }
        return ret;
    }

    Matrix getRowMatrix(const size_t rowIndex) const {
        Matrix ret(1, m_cols);

        for (size_t j = 0; j < m_cols; ++j) {
            ret(0, j) = m_data[rowIndex][j];
        }
        return ret;
    }

    Matrix getColMatrix(const size_t colIndex) const {
        Matrix ret(m_rows, 1);

        for (size_t i = 0; i < m_rows; ++i) {
            ret(i, 0) = m_data[i][colIndex];
        }
        return ret;
    }

    std::vector<std::vector<T>> transVec() {
        std::vector<std::vector<T>> ret(m_rows, std::vector<T>(m_cols));
        
        for (size_t i = 0; i < m_rows; ++i) {
            memcpy(&ret[i][0], &m_data[i][0], m_cols * sizeof(T));
        }
    }


    T& operator () (const size_t row, const size_t col) {
        return m_data[row][col];
    }

    template <size_t rows, size_t cols>
    Matrix& operator = (T (&Array)[rows][cols]) {
        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
                m_data[i][j] = Array[i][j];
            }
        }
        return *this;
    }

    Matrix& operator = (const Matrix& _matrix) {
        if (this == &_matrix) {
            return *this;
        }

        if (m_rows != _matrix.rows() || m_cols != _matrix.cols()) {
            //for (size_t i = 0; i < m_rows; ++i) {
            //    delete[] m_data[i];
            //}
            //delete[] m_data;
            delete[] * m_data, delete[] m_data;

            m_rows = _matrix.rows();
            m_cols = _matrix.cols();

            m_data = new T * [m_rows];
            for (size_t i = 0; i < m_rows; i++) {
                m_data[i] = new T[m_cols];
            }
        }

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; j++) {
                m_data[i][j] = _matrix.m_data[i][j];
            }
        }

        return *this;
    }

    Matrix& operator += (const Matrix& _matrix) {

        ASSERT(A.rows() == B.rows() || A.cols() == B.cols(), "Both matrix must have the same dimensions.");

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; j++) {
                m_data[i][j] += _matrix.m_data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator += (T number) {

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; j++) {
                m_data[i][j] += number;
            }
        }
        return *this;
    }

    Matrix& operator -= (const Matrix& _matrix){

        ASSERT(A.rows() == B.rows() || A.cols() == B.cols(), "Both matrix must have the same dimensions.");

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] -= _matrix.m_data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator -= (T number) {

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] -= number;
            }
        }
        return *this;
    }

    Matrix& operator *= (const Matrix& _matrix) {

        ASSERT(m_cols == _matrix.rows(), "Matrix inner dimensions must be agree.");

        Matrix temp(m_rows, _matrix.cols());
        for (size_t i = 0; i < temp.m_rows; ++i) {
            for (size_t j = 0; j < temp.m_cols; ++j) {
                for (size_t k = 0; k < m_cols; ++k) {
                    temp.m_data[i][j] += (m_data[i][k] * _matrix.m_data[k][j]);
                }
            }
        }
        return (*this = temp);
    }

    Matrix& operator *= (T number) {
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] *= number;
            }
        }
        return *this;
    }

    Matrix& operator /= (T number) {
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] /= number;
            }
        }
        return *this;
    }


    Matrix operator ^ (int number){
        Matrix temp(*this);
        return expHelper(temp, number);
    }

    Matrix operator + (const Matrix& _matrix) {
        Matrix temp(*this);
        return (temp += _matrix);
    }

    Matrix operator + (T number) {
        Matrix temp(*this);
        return (temp += number);
    }

    Matrix operator - (const Matrix& _matrix) {
        Matrix temp(*this);
        return (temp -= _matrix);
    }

    Matrix operator - (T number) {
        Matrix temp(*this);
        return (temp -= number);
    }

    Matrix operator * (const T number) {
        Matrix temp(*this);
        return (temp *= number);
    }

    Matrix operator * (const Matrix& _matrix) {
        Matrix temp(*this);
        return (temp *= _matrix);
    }

    Matrix operator / (const T number) {
        Matrix temp(*this);
        return (temp /= number);
    }
    
    bool operator == (const Matrix& _matrix) {
        if (m_size != _matrix.size() || m_rows != _matrix.rows() || m_cols != _matrix.cols())
            return false;

        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; j++) {
                if (m_data[i][j] != _matrix.m_data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }   

    bool operator != (const Matrix& _matrix) {
        return !(*this == _matrix);
    }

    friend std::ostream& operator << (std::ostream& os, const Matrix& _matrix) {
        for (size_t i = 0; i < _matrix.rows(); ++i) {
            os << _matrix.m_data[i][0];
            for (size_t j = 1; j < _matrix.cols(); ++j) {
                os << " " << _matrix.m_data[i][j];
            }
            os << std::endl;
        }
        return os;
    }

    friend std::istream& operator >> (std::istream& is, const Matrix& _matrix) {
        for (size_t i = 0; i < _matrix.rows(); ++i) {
            for (size_t j = 0; j < _matrix.cols(); ++j) {
                is >> _matrix.m_data[i][j];
            }
        }
        return is;
    }

    
    Matrix transpose() {
        Matrix ret(m_cols, m_rows);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                ret.m_data[j][i] = m_data[i][j];
            }
        }
        return ret;
    }

    Matrix inverse() {
        Matrix I = Identity(m_rows);
        Matrix AI = augment(*this, I);
        Matrix U = AI.gaussianEliminate();
        Matrix IAInverse = U.rowReduceFromGaussian();
        Matrix AInverse(m_rows, m_cols);
        for (size_t i = 0; i < AInverse.rows(); ++i) {
            for (size_t j = 0; j < AInverse.cols(); ++j) {
                AInverse(i, j) = IAInverse(i, j + m_cols);
            }
        }
        return AInverse;
    }

    void swapRows(const size_t r1, const size_t r2)
    {
        T* temp = m_data[r1];
        m_data[r1] = m_data[r2];
        m_data[r2] = temp;
    }


    static T dot(const Matrix& A, const Matrix& B) {

        ASSERT(A.rows() == B.rows() || A.cols() == B.cols(), "Both matrix must have the same dimensions. So return a empty matrix.");

        T sum = 0;
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                sum += (A(i, j) * B(i, j));
            } 
        }
        return sum;
    }

    static Matrix mul(const Matrix& A, const Matrix& B) {

        ASSERT(A.rows() == B.rows() || A.cols() == B.cols(), "Both matrix must have the same dimensions.");

        Matrix temp(A.rows(), A.cols());
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                temp(i, j) = A.m_data[i][j] * B.m_data[i][j];
            }
        }
        return temp;
    }


    Matrix gaussianEliminate()
    {
        Matrix Ab(*this);
        int rows = Ab.rows();
        int cols = Ab.cols();
        int Acols = cols - 1;

        int i = 0; 
        int j = 0; 

        while (i < rows)
        {
            bool pivot_found = false;
            while (j < Acols && !pivot_found)
            {
                if (Ab(i, j) != 0) { 
                    pivot_found = true;
                }
                else {
                    int max_row = i;
                    double max_val = 0;
                    for (int k = i + 1; k < rows; ++k)
                    {
                        double cur_abs = Ab(k, j) >= 0 ? Ab(k, j) : -1 * Ab(k, j);
                        if (cur_abs > max_val)
                        {
                            max_row = k;
                            max_val = cur_abs;
                        }
                    }
                    if (max_row != i) {
                        Ab.swapRows(max_row, i);
                        pivot_found = true;
                    }
                    else {
                        j++;
                    }
                }
            }

            if (pivot_found)
            {
                for (int t = i + 1; t < rows; ++t) {
                    for (int s = j + 1; s < cols; ++s) {
                        Ab(t, s) = Ab(t, s) - Ab(i, s) * (Ab(t, j) / Ab(i, j));
                        if (Ab(t, s) < EPS && Ab(t, s) > -1 * EPS)
                            Ab(t, s) = 0;
                    }
                    Ab(t, j) = 0;
                }
            }

            i++;
            j++;
        }

        return Ab;
    }

    Matrix rowReduceFromGaussian()
    {
        Matrix R(*this);
        int rows = R.rows();
        int cols = R.cols();

        int i = rows - 1; 
        int j = cols - 2; 

        while (i >= 0)
        {
            int k = j - 1;
            while (k >= 0) {
                if (R(i, k) != 0)
                    j = k;
                k--;
            }

            if (R(i, j) != 0) {

                for (int t = i - 1; t >= 0; --t) {
                    for (int s = 0; s < cols; ++s) {
                        if (s != j) {
                            R(t, s) = R(t, s) - R(i, s) * (R(t, j) / R(i, j));
                            if (R(t, s) < EPS && R(t, s) > -1 * EPS)
                                R(t, s) = 0;
                        }
                    }
                    R(t, j) = 0;
                }

                for (int k = j + 1; k < cols; ++k) {
                    R(i, k) = R(i, k) / R(i, j);
                    if (R(i, k) < EPS && R(i, k) > -1 * EPS)
                        R(i, k) = 0;
                }
                R(i, j) = 1;

            }

            i--;
            j--;
        }

        return R;
    }

    static Matrix augment(const Matrix& A, const Matrix& B)
    {
        Matrix AB(A.rows(), A.cols() + B.cols());
        for (int i = 0; i < AB.rows(); ++i) {
            for (int j = 0; j < AB.cols(); ++j) {
                if (j < A.cols())
                    AB(i, j) = A(i, j);
                else
                    AB(i, j) = B(i, j - B.cols_);
            }
        }
        return AB;
    }

    static Matrix LUDecomposition(const Matrix& A, const Matrix& B) {
        Matrix LU(A);

        size_t m = A.rows();
        size_t n = A.cols();
        
        std::vector<size_t> piv(m);
        for (size_t i = 0; i < m; i++) {
            piv[i] = i;
        }

        int pivsign = 1;
        T LUrowi[n];
        T LUcolj[m];
        for (size_t j = 0; j < n; ++j) {

            for (size_t i = 0; i < m; ++i) {
                LUcolj[i] = LU(i, j);
            }
            for (size_t i = 0; i < m; ++i) {

                for (size_t k = 0; k < n; ++n) {
                    LUrowi[k] = LU(i, k);
                }
                size_t kmax = (std::min)(i, j);
                auto s = 0;
                for (size_t k = 0; k < kmax; ++k) {
                    s += LUrowi[k] * LUcolj[k];
                }
                LUrowi[j] = LUcolj[i] -= s;
            }

            size_t p = j;
            for (size_t i = j + 1; i < m; ++i) {
                if ((std::abs)(LUcolj[i]) > (std::abs)(LUcolj[p])) {
                    p = i;
                }
            }

            if (p != j) {
                for (size_t k = 0; k < n; ++k) {
                    auto t = LU[p][k];
                    LU(p, k) = LU(j, k);
                    LU(j, k) = t;
                }
                auto k = piv[p];
                piv[p] = piv[j];
                piv[j] = k;
                pivsign = -pivsign;
            }

            if (j < m && LU(j, j) != 0) {
                for (size_t i = j + 1; i < m; ++i) {
                    LU(i, j) /= LU(j, j);
                }
            }
        }

        ASSERT(B.rows() == m, "Matrix row dimensions must agree.");
        for (size_t j = 0; j < n; j++) {
            ASSERT(LU(j, j) != 0, "Matrix is singular.");
        }

        size_t nx = B.cols();
        Matrix X = B.getMatrix(piv, 0, nx - 1); 

        for (size_t k = 0; k < n; ++k) {
            for (size_t i = k + 1; i < n; ++i) {
                for (size_t j = 0; j < nx; ++j) {
                    X(i, j) -= X(k, j) * LU(i, k);
                }
            }
        }

        for (size_t k = n - 1; k >= 0; --k) {
            for (size_t j = 0; j < nx; ++j) {
                X(k, j) /= LU(k, k);
            }
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = 0; j < nx; ++j) {
                    X(i, j) -= X(k, j) * LU(i, k);
                }
            }
        }
        return X;
    }

    static Matrix QRDecomposition(const Matrix& A, const Matrix& B) {
        Matrix QR(A);

        size_t m = A.rows();
        size_t n = A.cols();

        T Rdiag[n];
        T nrm;

        for (size_t k = 0; k < n; ++k) {
            nrm = 0;
            for (size_t i = k; i < m; ++i) {
                nrm = (std::hypot)(nrm, QR(i, k)); 
            }
            if (nrm != 0) {
                if (QR(k, k) < 0) {
                    nrm = -nrm;
                }
                for (size_t i = k; i < m; ++i) {
                    QR(i, k) /= nrm;
                }
                QR(k, k) += 1;

                for (size_t j = k + 1; j < n; ++j) {
                    T s = 0;
                    for (size_t i = k; i < m; ++i) {
                        s += QR(i, k) * QR(i, j);
                    }
                    s = -s / QR(k, k);
                    for (size_t i = k; i < m; ++i) {
                        QR(i, j) += s * QR(i, k);
                    }
                }
            }
            Rdiag[k] = -nrm;
        }

        ASSERT(B.rows() == m, "Matrix row dimensions must agree.");

        for (size_t j = 0; j < n; ++j) {
            ASSERT(Rdiag[j] != 0, "Matrix is rank deficient.");
        }

        size_t nx = B.cols();
        Matrix X(B);

        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < nx; ++j) {
                T s = 0.0;                  
                for (size_t i = k; i < m; ++i) {
                    s += QR(i, k) * X(i, j);
                }
                s = -s / QR(k, k);
                for (size_t i = k; i < m; ++i) {
                    X(i, j) += s * QR(i, k);
                }
            }
        }

        for (size_t k = n - 1; k >= 0; --k) {
            for (size_t j = 0; j < nx; ++j) {
                X(k, j) /= Rdiag[k];
            }
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = 0; j < nx; ++j) {
                    X(i, j) -= X(k, j) * QR(i, k);
                }
            }
        }
        return X.getSubMatrix(0, n - 1, 0, nx - 1);
    }

private:

    Matrix expHelper(const Matrix& _matrix, int number)
    {
        if (number == 0) {
            return Identity(_matrix.rows());
        }
        else if (number == 1) {
            return _matrix;
        }
        else if (number % 2 == 0) {  // num is even
            return expHelper(_matrix * _matrix, number / 2);
        }
        else {                    // num is odd
            return _matrix * expHelper(_matrix * _matrix, (number - 1) / 2);
        }
    }

private:
    size_t  m_rows;
    size_t  m_cols;
    size_t  m_size;
    T**     m_data;
};

#endif 


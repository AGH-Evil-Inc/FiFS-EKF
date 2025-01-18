# GyroAccKalmanFilter
Trying to get orientation of an object based on accelerometer and qyroscope measurments

## 1. Kalibracja czujników

### GYRO (żyroskop):
- Wyznaczenie biasów początkowych - wprowadzane do EKF, zmieniane wewnątrz, nie są odejmowane przed,
- Wyznaczenie wariancji dla każdej osi - wprowadzane do EKF
### ACC (akcelerometr):
Wyznaczenie macierzy kalibracyjnej (skala i biasy):
$$
\underbrace{
\begin{bmatrix}
x_{\text{calibrated}} \\
y_{\text{calibrated}} \\
z_{\text{calibrated}}
\end{bmatrix}
}_{b_{calibrated}}
=
\underbrace{
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
}_{C_{params}}
\underbrace{
\begin{bmatrix}
x_{\text{raw}} \\
y_{\text{raw}} \\
z_{\text{raw}}
\end{bmatrix}
}_{b_{raw}}
+
\underbrace{
\begin{bmatrix}
j \\
k \\
l
\end{bmatrix}
}_{b_{biases}}
$$
...przepisane (metoda na uzyskiwanie wartości skalibrowanych z surowych danych, pojedynczym przemnożeniem):
$$
\underbrace{
\begin{bmatrix}
x_{\text{calibrated}} & y_{\text{calibrated}} & z_{\text{calibrated}}
\end{bmatrix}
}_{b_{calibrated}}
=
\underbrace{
\begin{bmatrix}
x_{\text{raw}} & y_{\text{raw}} & z_{\text{raw}} & 1
\end{bmatrix}
}_{b_{raw} \& 1}
\underbrace{
\begin{bmatrix}
a & d & g \\
b & e & h \\
c & f & i \\
j & k & l
\end{bmatrix}
}_{X = C_{params} \& b_{biases}}
$$

$$x_{true} := x_{calibrated}$$

$$
R = \begin{bmatrix}
x_{\text{raw}1} & y_{\text{raw}1} & z_{\text{raw}1} & 1 \\
x_{\text{raw}2} & y_{\text{raw}2} & z_{\text{raw}2} & 1 \\
\vdots & \vdots & \vdots & \vdots \\
x_{\text{raw}N} & y_{\text{raw}N} & z_{\text{raw}N} & 1
\end{bmatrix}
$$

$$
T = \begin{bmatrix}
x_{\text{true}1} & y_{\text{true}1} & z_{\text{true}1} \\
x_{\text{true}2} & y_{\text{true}2} & z_{\text{true}2} \\
\vdots & \vdots & \vdots \\
x_{\text{true}N} & y_{\text{true}N} & z_{\text{true}N}
\end{bmatrix}
$$

$$T = R \cdot X$$
$$R \cdot X = T$$

Ostateczne, wyznaczenie macierzy kalibracji $X$:
$$X = \left(R^T \cdot R\right)^{-1} \cdot R^T \cdot T$$


### MAG (magnetometr)
- **Na razie nieużyty w projekcie!** - brak dobrych danych kalibracyjnych (TODO)

Gdyby był użyty - kalibracja metodą optymalizacji parametrycznej:
- model pomiaru: $B_{\text{mierz}} = C \cdot B_{\text{rzecz}} + b$:
  - $B_{rzecz}$ - rzeczywista wartość pola magnetycznego,
  - $C$ - macierz kalibracji,
  - $b$ - biasy magnetometru.


## 2. Struktura filtru

### 2.1. Wektor stanu $x$ - równanie

$
\begin{align}
x =
\begin{bmatrix}
q_w \\
q_x \\
q_y \\
q_z \\
b_x \\
b_y \\
b_z
\end{bmatrix}
\end{align}
$

Wektor ma wymiary $7 \times 1$, a każda z wartości jest jednowymiarowa. Wartości oznaczone przez $q$ to jednostki kwaternionu opisującego aktualny obrót badanego obiektu w przestrzeni, a wartości oznaczone przez $b$ to biasy żyroskopu w każdej z mierzonych osi. Teoretycznie wektor mógłby być dwuelementowy, zawierający czterowymiarowy kwaternion i trójwymiarowy wektor biasu.

### 2.2. Nieliniowy model dynamiki $f(x,\omega)$:

Predykcja stanu aktualizuje tylko 4 pierwsze elementy wektora stanu (kwaterniona rotacji) i odbywa się w następujący sposób:

Na podstawie prędkości kątowej ($\omega$) odczytanej z żyroskopu, oraz biasów ($b$) zawartych w 3 ostatnich elementach wektora stanu ($x$), wyznaczamy zmianę wektora rotacji w danej jednostce czasu.

$
\begin{align}
\Delta \vec{\theta} = (\vec{\omega} - \vec{b}) \Delta t
\end{align}
$

Zmieniamy reprezentację wektora zmiany rotacji na postać kwaternionu.

$
\begin{align}
\hat{u} = \frac{\Delta \vec{\theta}}{|\Delta \vec{\theta}|}
\end{align}
$

$
\begin{align}
\Delta q = 
\begin{bmatrix}
\cos(\frac{|\Delta \vec{\theta}|}{2}), \hat{u} \sin(\frac{|\Delta \vec{\theta}|}{2})
\end{bmatrix}
\end{align}
$

Przewidujemy stan "obracając" aktualny kwaternion rotacji przez wyznaczoną powyżej zmianę ($\otimes$ - mnożenie kwaternionów). W praktyce po prostu nacałkowujemy prędkość kątową odczytaną z żyroskopu. $q$ to kwaternion złożony z 4 pierwszych elementów wektora stanu.

$
\begin{align}
q_{k+1} = q_k \otimes \Delta q
\end{align}
$

### 2.3. Macierz przejścia stanu $F$ (linearyzacja):

W przypadku bardzo małych zmian kąta $\theta$ zmianę kwaternionu można przybliżyć.

$
\begin{align}
\Delta q \approx 
\begin{bmatrix}
1, \frac{|\Delta \vec{\theta}|}{2} \hat{u}
\end{bmatrix}
\end{align}
$

Otrzymujemy w ten sposób funkcję przejścia stanu.

$
\begin{align}
f(x, \omega, w) = 
\begin{bmatrix}
q_w - \frac{\Delta t}{2} q_x (\omega_x - b_x) - \frac{\Delta t}{2} q_y (\omega_y - b_y) - \frac{\Delta t}{2} q_z (\omega_z - b_z) \\
q_x + \frac{\Delta t}{2} q_w (\omega_x - b_x) - \frac{\Delta t}{2} q_z (\omega_y - b_y) + \frac{\Delta t}{2} q_y (\omega_z - b_z) \\
q_y + \frac{\Delta t}{2} q_z (\omega_x - b_x) + \frac{\Delta t}{2} q_w (\omega_y - b_y) - \frac{\Delta t}{2} q_x (\omega_z - b_z) \\
q_z - \frac{\Delta t}{2} q_y (\omega_x - b_x) + \frac{\Delta t}{2} q_x (\omega_y - b_y) + \frac{\Delta t}{2} q_w (\omega_z - b_z) \\
b_x \\
b_y \\
b_z
\end{bmatrix} + w
\end{align}
$

Macierz przejścia stanu F jest jakobianem powyższej funkcji.

$
\begin{align}
F =
\begin{bmatrix}
1 & \frac{\Delta t}{2}(-\omega_x + b_x) & \frac{\Delta t}{2}(-\omega_y + b_y) & \frac{\Delta t}{2}(-\omega_z + b_z) & \frac{\Delta t q_x}{2} & \frac{\Delta t q_y}{2} & \frac{\Delta t q_z}{2} \\
\frac{\Delta t}{2}(\omega_x - b_x) & 1 & \frac{\Delta t}{2}(\omega_z - b_z) & \frac{\Delta t}{2}(-\omega_y + b_y) & -\frac{\Delta t q_w}{2} & \frac{\Delta t q_z}{2} & -\frac{\Delta t q_y}{2} \\
\frac{\Delta t}{2}(\omega_y - b_y) & \frac{\Delta t}{2}(-\omega_z + b_z) & 1 & \frac{\Delta t}{2}(\omega_x - b_x) & -\frac{\Delta t q_z}{2} & -\frac{\Delta t q_w}{2} & \frac{\Delta t q_x}{2} \\
\frac{\Delta t}{2}(\omega_z - b_z) & \frac{\Delta t}{2}(\omega_y - b_y) & \frac{\Delta t}{2}(-\omega_x + b_x) & 1 & \frac{\Delta t q_y}{2} & -\frac{\Delta t q_x}{2} & -\frac{\Delta t q_w}{2} \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
\end{align}
$

### 2.4. Model pomiarowy $h(x)$ w filtrze

$
\begin{align}
h(x, v) =
\begin{bmatrix}
-2g ( q_x q_z - q_w q_y ) \\
-2g ( q_y q_z + q_w q_x ) \\
-g ( q_w^2 - q_x^2 - q_y^2 + q_z^2 )
\end{bmatrix} + v
\end{align}
$

Gdzie jednostki $q_w, q_x, q_y, q_z$ to 4 pierwsze elementy wektora stanu $x$, a g to stała grawitacji.

Naszą obserwacją jest wektor będący przewidywanym teoretycznym odczytem pomiaru z akcelerometru. Powyższy wzór jest rezultatem zamiany kwaternionu stanu na macierz rotacji i użycie jej do konwersji wektora grawitacji z globalnego do lokalnego układu odniesienia.

### 2.5. Macierz modelu pomiarowego $H$ (linearyzacja):

$
\begin{align}
H = 2g
\begin{bmatrix}
q_y & -q_z & q_w & -q_x & 0 & 0 & 0 \\
-q_x & -q_w & -q_z & -q_y & 0 & 0 & 0 \\
-q_w & q_x & q_y & -q_z & 0 & 0 & 0
\end{bmatrix}
\end{align}
$

Macierz H jest jakobianem funkcji obserwacji h.


## 3. Warunki początkowe i parametry filtru

### 3.1. Początkowa wartość estymaty stanu

**MOŻLIWOŚĆ ZMIANY:**  
W przypadku dostępności dobrych danych do kalibracji MAG (magnetometru), do wyznaczenia orientacji początkowej mógłby zostać wykorzystany MAG w połączeniu z ACC - np. używając **algorytmu TRIAD** (z wykładu) lub innego.

W implementacji początkowa orientacja to $(0, 0, 0)$ (kąty Eulera), czyli $(1, 0, 0, 0)$ (kwaternion). 
Wynika to z dostosowania implementacji do zestawu danych testowych od prowadzącego, gdzie przez pierszy okres orientacja jest stała, równa $(0, 0, 0)$. Operując w środowisku eksperymentu, podjęto również decyzję o braku estymacji początkowych Roll i Pitch (z Yaw ustawionym na 0) na podstawie samego ACC, gdyż mogłoby to niepotrzebnie zaburzyć działanie EKF, gdzie połączenie ACC i MAG już by zostało zaimplementowane.

$
\begin{align}
\hat{x}_0 =
\begin{bmatrix}
q_w = 1 \\
q_x = 0 \\
q_y = 0\\
q_z = 0\\
{b_x}_0 \\
{b_y}_0 \\
{b_z}_0
\end{bmatrix}
\end{align}
$


### Początkowa wartość estymaty kowariancji stanu $P_0$

Podobnie jak wyżej - z powodu dostosowania do zestawu danych tekstowych z eksperymentu, gdzie na początku czujniki się nie poruszają - ustawiono kowariancje na $0$, ponieważ po kilku krokach czasu na wartości macierzy kowariancji $P$ wpłynęły wartości szumów z $Q$ i $Q_{bias}$

### Wartość macierzy kowariancji procesu $Q$

Szum procesu jest kombinacją szumu pomiarowego żyroskopu oraz szumu biasu żyroskopu. Szum pomiarowy żyroskopu znamy w postaci szumu każdej z osi, więc macierz $Q_{\text{gyro}}$ musimy przekształcić (za pomocą macierzy $W$) na postać odpowiadającą naszej macierzy stanu - na 
macierz kowariancji jednostek kwaternionu. Szum procesu wyznaczamy więc za pomocą poniższego wzoru:

$
\begin{align}
Q = WQ_{\text{gyro}}W^T + Q_{\text{bias}}
\end{align}
$

Gdzie:

$
\begin{align}
Q_{\text{gyro}} =
\begin{bmatrix}
\sigma_{gx}^2 & 0 & 0 \\
0 & \sigma_{gy}^2 & 0 \\
0 & 0 & \sigma_{gz}^2
\end{bmatrix}
\end{align}
$

$
\begin{align}
W = \frac{\Delta t}{2}
\begin{bmatrix}
-q_x & -q_y & -q_z \\
q_w & -q_z & q_y \\
q_z & q_w & -q_x \\
-q_y & q_x & q_w \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\end{align}
$

$
\begin{align}
Q_{\text{bias}} =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \sigma_{bx}^2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & \sigma_{by}^2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & \sigma_{bz}^2
\end{bmatrix}
\end{align}
$

Gdzie $\sigma_{gx}^2, \sigma_{gy}^2, \sigma_{gz}^2$ to wariancje szumu każdej osi żyroskopu, a $\sigma_{bx}^2, \sigma_{by}^2, \sigma_{bz}^2$ to wariancje biasów każdej osi żyroskopu.

### Wartość macierzy kowariancji pomiarów $R$

$
\begin{align}
R =
\begin{bmatrix}
\sigma_{ax}^2 & 0 & 0 \\
0 & \sigma_{ay}^2 & 0 \\
0 & 0 & \sigma_{az}^2
\end{bmatrix}
\end{align}
$

Gdzie $\sigma_{ax}^2, \sigma_{ay}^2, \sigma_{az}^2$ to wariancje szumu każdej osi akcelerometru.


## Wyniki

TODO


## Bibliografia

- [Cookie Robotics: Quaternion-Based EKF for Attitude and Bias Estimation](https://cookierobotics.com/073/):
  - implementacja EKF z GYRO i ACC (bez MAG),
  - wektor stanu $x$ złożony z orientacji w formie kwaternionu i biasów GYRO,
  - predykcja stanu w formie wektorowego wyznaczania $\Delta{}q$,
- [Cookie Robotics: Accelerometer Calibration - 6 point calibration](https://cookierobotics.com/061/)
  - do ew. doimplementowania
- [Cookie Robotics: Accelerometer Calibration Methods](https://cookierobotics.com/064/)
- [AHRS: Extended Kalman Filter](https://ahrs.readthedocs.io/en/latest/filters/ekf.html):
  - implementacja EKF z GYRO, ACC, MAG,
  - wektor stanu $x$ złożony tylko z orientacji w formie kwaternionu
# Modul Pembelajaran: Model Concept dalam Machine Learning

Dokumentasi pembelajaran ini membahas konsep paling mendasar dalam rekayasa *Machine Learning* (ML): **apa sebenarnya yang dimaksud dengan "model"**, bagaimana ia terbentuk, komponen apa saja yang menyusunnya, serta bagaimana model memandang dan memisahkan data di dunia nyata.

---

## Glosarium Istilah Penting
Agar tidak membingungkan, berikut arti istilah teknis yang digunakan dalam modul ini:
* **Fitur Masukan (*Features*, $X$)**: karakteristik atau informasi yang menjadi bahan pertimbangan komputer untuk mengambil keputusan (misalnya: luas tanah, jumlah kamar, dan lokasi).
* **Target yang Ingin Diprediksi (*Target / Label*, $Y$)**: nilai atau jawaban akhir yang ingin ditebak oleh model (misalnya: harga rumah).
* **Parameter Dalam (*Model Parameters*)**: nilai internal di dalam model yang dicari dan disesuaikan sendiri oleh algoritma secara otomatis dari data latih saat proses belajar.
* **Setelan Luar (*Hyperparameters*)**: tombol konfigurasi di luar model yang ditentukan secara manual oleh manusia perekayasa ML sebelum proses belajar dimulai.
* **Ruang Hipotesis (*Hypothesis Space*, $\mathcal{H}$)**: kumpulan seluruh kemungkinan rumus matematika atau bentuk fungsi yang diizinkan untuk dicoba oleh algoritma.
* **Batas Keputusan (*Decision Boundary*)**: garis, kurva, atau bidang pemisah di ruang data yang membedakan tebakan kelas satu dengan kelas lainnya.
* **Generalisasi (*Generalization*)**: kemampuan model untuk memberikan tebakan yang akurat pada data baru yang belum pernah dipelajari sebelumnya.

---

## 1. Apa Sebenarnya "Model" Itu? (*What is a Model?*)

Di dunia kecerdasan buatan, istilah **Algoritma** dan **Model** sering tertukar, padahal keduanya memiliki peran yang berbeda.

### 1.1 Analogi Koki, Bahan Masakan, dan Resep Jadi
Bayangkan sebuah proses memasak di dapur:
* **Data Latih** adalah **bahan masakan mentah** (sayur, bumbu, daging).
* **Algoritma Machine Learning** adalah **koki atau prosedur memasak** (teknik menumis, merebus, atau memanggang).
* **Model Machine Learning** adalah **resep masakan jadi** yang sudah matang dan teruji takarannya.

Ketika restoran membuka cabang baru di tempat lain, pemilik restoran tidak perlu membawa seluruh bahan makanan lama. Mereka hanya perlu membawa **resep jadi (model)** tersebut agar koki lain bisa menyajikan hidangan dengan cita rasa yang konsisten.

```
+-----------------------------------------------------------------------------+
|                           PROSES PEMBENTUKAN MODEL                          |
+-----------------------------------------------------------------------------+

  [ Data Masukan (X) ] 
          +             ---> [ Algoritma Belajar ] ---> [ MODEL ML ]
  [ Target Sejati (Y)]       (Proses Optimasi)          (Fungsi Matematis f(X))
                                                               |
                                                               v
                                                  Saat Digunakan di Produksi:
                                                  Data Baru ---> [ MODEL ] ---> Hasil Tebakan
```

### 1.2 Definisi Formal Secara Matematis
Secara matematis, model adalah sebuah **fungsi aproksimasi (pendekatan)**:

$$\hat{Y} = f(X; \theta)$$

* $X$ adalah vektor data masukan.
* $\theta$ (theta) adalah kumpulan angka parameter dalam yang sudah terkunci nilainya.
* $f$ adalah struktur logika atau rumus dari model.
* $\hat{Y}$ adalah hasil tebakan yang dikeluarkan oleh model.

Model bukanlah makhluk yang memiliki kesadaran, melainkan representasi matematis kompak yang merangkum pola dari masa lalu untuk menerka apa yang akan terjadi pada data baru.

---

### 1.3 Sumber Rujukan Topik 1
* **Mitchell, Tom M. (1997)**, *Machine Learning*, McGraw-Hill Computer Science Series, Bab 1: “Introduction”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “The Machine Learning Landscape”.

---

## 2. Parameter vs Hyperparameter

Perbedaan antara *parameter* dan *hyperparameter* terletak pada **siapa yang menentukan nilainya** dan **kapan nilai tersebut ditetapkan**.

```
                           KOMPARASI DUA SETELAN ML
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
  PARAMETER DALAM                                           SETELAN LUAR
 (Model Parameters)                                      (Hyperparameters)
- Ditemukan otomatis oleh komputer.                     - Disetel manual oleh manusia.
- Berubah sepanjang proses latihan.                     - Dikunci sebelum latihan dimulai.
- Contoh: Bobot (w) dan Bias (b).                       - Contoh: Learning rate, kedalaman pohon.
```

---

### 2.1 Parameter Dalam (*Model Parameters*)
Parameter adalah variabel internal yang nilainya diestimasi secara mandiri oleh algoritma menggunakan data latih.

* **Cara Kerja**: pada awal latihan, parameter biasanya diberi angka acak atau nol. Setiap kali model melakukan kesalahan tebak, algoritma optimasi (seperti *Gradient Descent*) akan menggeser angka parameter ini sedikit demi sedikit agar kesalahan tebakannya mengecil.
* **Contoh Nyata**:
  * Pada Regresi Linear: nilai kemiringan garis (*slope* atau bobot $w$) dan titik potong sumbu (*intercept* atau bias $b$) pada rumus $y = wx + b$.
  * Pada Jaringan Saraf Tiruan (*Neural Networks*): jutaan bobot koneksi antarsaraf tiruan.
* Nilai parameter inilah yang **disimpan ke dalam berkas model** saat proyek selesai dilatih.

---

### 2.2 Setelan Luar (*Hyperparameters*)
Hyperparameter adalah setelan konfigurasi di luar tubuh model yang mengatur **bagaimana cara algoritma tersebut belajar**.

* **Cara Kerja**: komputer tidak bisa menentukan sendiri setelan ini dari rumus dasar. Perekayasa perangkat lunak harus menentukannya berdasarkan intuisi, eksperimen, atau pencarian otomatis (*Grid Search* / *Random Search*).
* **Contoh Nyata**:
  * **Laju Belajar (*Learning Rate*, $\alpha$)**: seberapa besar langkah perubahan bobot di setiap putaran belajar.
  * **Nilai $k$ pada k-Nearest Neighbors**: berapa banyak tetangga terdekat yang diajak bermusyawarah untuk menentukan tebakan kelas.
  * **Kedalaman Maksimum (*max_depth*) pada Decision Tree**: berapa tingkat percabangan pertanyaan yang boleh dibuat oleh pohon keputusan.
  * **Jumlah Pengulangan (*Epochs*)**: berapa kali model diizinkan membaca ulang seluruh dataset latihan.

---

### 2.3 Tabel Komparasi Rinci
| Dimensi Evaluasi | Parameter Dalam (*Parameters*) | Setelan Luar (*Hyperparameters*) |
| :--- | :--- | :--- |
| **Pihak Penentu** | Algoritma optimasi secara mandiri | Manusia / perekayasa data |
| **Waktu Penentuan** | Selama proses latihan berjalan | Sebelum proses latihan dimulai |
| **Sumber Nilai** | Dihitung langsung dari data latih | Diuji coba lewat data validasi |
| **Penyimpanan Berkas** | Disimpan permanen sebagai isi model | Disimpan dalam konfigurasi kode program |
| **Tujuan Utama** | Menangkap pola spesifik dari data | Mengontrol kapasitas dan kestabilan latihan |

---

### 2.4 Sumber Rujukan Topik 2
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 5: “Machine Learning Basics (Parameters and Hyperparameters)”.
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 7: “Model Assessment and Selection”.

---

## 3. Intuisi Ruang Hipotesis (*Hypothesis Space Intuition*)

Ketika kita memilih sebuah algoritma machine learning, kita sebenarnya sedang menentukan **kacamata** seperti apa yang akan digunakan komputer untuk memandang dunia data.

### 3.1 Definisi Ruang Hipotesis ($\mathcal{H}$)
**Ruang Hipotesis** adalah himpunan seluruh kemungkinan fungsi matematika atau bentuk solusi yang secara teoretis dapat dibentuk oleh suatu algoritma.

* **Analogi Cetakan Kue**:
  * Jika kamu memilih algoritma Regresi Linear, kamu memberikan komputer **cetakan berbentuk penggaris lurus**.
  * Ruang hipotesisnya adalah **semua posisi garis lurus di dunia** (garis miring ke atas, datar, miring curam, dan seterusnya).
  * Komputer bertugas mencari satu garis lurus paling pas di antara miliaran pilihan garis lurus yang ada di ruang hipotesis tersebut.
  * Namun, sekeras apa pun komputer belajar, ia **tidak akan pernah bisa menghasilkan garis melengkung berbentuk lingkaran**, karena bentuk melengkung berada di luar ruang hipotesis yang kamu sediakan.

```
+-----------------------------------------------------------------------------+
|                     ILUSTRASI RUANG HIPOTESIS (H)                           |
+-----------------------------------------------------------------------------+

  [ Seluruh Kemungkinan Fungsi di Alam Semesta ]
  +---------------------------------------------------------------+
  |                                                               |
  |   Ruang Hipotesis Model Linear (H_linear)                     |
  |   +---------------------------------------+                   |
  |   | Garis datar, garis miring curam,      |                   |
  |   | garis miring landai (Hanya Garis)     |                   |
  |   +---------------------------------------+                   |
  |                                                               |
  |   Ruang Hipotesis Model Polinomial / Non-Linear (H_kompleks)  |
  |   +-------------------------------------------------------+   |
  |   | Kurva parabola, gelombang S, lingkaran, batas bebas   |   |
  |   +-------------------------------------------------------+   |
  |                                                               |
  |   * Fungsi Sejati Pembentuk Data Dunia Nyata (f sejati)       |
  +---------------------------------------------------------------+
```

---

### 3.2 Bias Induktif (*Inductive Bias*)
Mengapa kita tidak selalu memakai algoritma dengan ruang hipotesis seluas mungkin? Karena jika model terlalu bebas tanpa batasan, model akan bingung dan menghafal seluruh derau (*noise*).

Batasan awal yang kita berikan kepada model disebut **Bias Induktif**:
* Bias induktif dari Regresi Linear: *"Saya berasumsi hubungan antardata di dunia ini berbentuk garis lurus proporsional."*
* Bias induktif dari Pohon Keputusan (*Decision Tree*): *"Saya berasumsi data dapat dipisahkan secara bertahap menggunakan garis tegak lurus sumbu (ortogonal)."*
* Bias induktif dari k-Nearest Neighbors: *"Saya berasumsi data yang letaknya berdekatan memiliki kelas atau karakteristik yang sama."*

### 3.3 Kaitan Ruang Hipotesis dengan Underfitting dan Overfitting
* **Ruang Hipotesis Terlalu Sempit (Bias Terlalu Kuat)**: model terlalu kaku. Walaupun data di lapangan melengkung, model memaksakan diri menarik garis lurus. Akibatnya terjadi **kurang pas (*underfitting*)**.
* **Ruang Hipotesis Terlalu Luas (Varians Terlalu Tinggi)**: model terlalu fleksibel dan berdaya tampung raksasa. Model membentuk lekukan ekstrem demi melewati setiap titik data latih. Akibatnya terjadi **menghafal mati (*overfitting*)**.

---

### 3.4 Sumber Rujukan Topik 3
* **Mitchell, Tom M. (1997)**, *Machine Learning*, McGraw-Hill Computer Science Series, Bab 2: “Concept Learning and the General-to-Specific Ordering (Inductive Bias)”.
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 1: “Introduction (Model Selection and the Curse of Dimensionality)”.

---

## 4. Batas Keputusan (*Decision Boundaries - Visual Intuition*)

Untuk masalah klasifikasi (misalnya membedakan transaksi sah vs penipuan), model bertugas menarik pembatas di dalam ruang fitur.

### 4.1 Definisi Batas Keputusan
**Batas Keputusan (*Decision Boundary*)** adalah garis, kurva, atau bidang batas di ruang data tempat model **berubah pikiran** dari satu kelas ke kelas lain.

Di sepanjang garis batas ini, model berada pada tingkat keraguan tertinggi: probabilitas data tergolong ke Kelas A atau Kelas B bernilai seimbang tepat $50\% : 50\%$.

---

### 4.2 Intuisi Visual Berbagai Batas Keputusan

Mari amati bagaimana berbagai jenis algoritma memisahkan dua kelompok data (tanda `O` dan tanda `X`) pada bidang dua dimensi:

#### A. Batas Garis Lurus (Model Linear: Logistic Regression / Linear SVM)
Memotong ruang data menggunakan satu garis lurus diagonal, datar, atau tegak tanpa lekukan:

```
Fitur 2 ^
        |      O      O   /   X      X
        |   O      O     /        X
        |      O        /      X      X
        |   O          /   X       X
        |             /  <--- Batas Garis Lurus Tunggal
        |            /
        +----------------------------------->
                                       Fitur 1
```

#### B. Batas Bertangga / Kotak-Kotak (Decision Tree)
Pohon keputusan memotong ruang fitur menggunakan serangkaian aturan kondisi bertingkat (`Fitur 1 > 5`, lalu `Fitur 2 < 3`). Batasnya selalu berbentuk garis patah-patah bersudut siku-siku ($90^\circ$):

```
Fitur 2 ^
        |   O      O   |   X      X
        |   O      O   |   X      X
        |   -----------+---------     <--- Potongan Siku-Siku
        |   O      O   |       X
        |   O      O   |   X      X
        +----------------------------------->
                                       Fitur 1
```

#### C. Batas Kurva Mulus (Kernel SVM / Neural Networks)
Mampu memisahkan data yang posisinya melingkar konsentris (misal data `O` dikelilingi cincin data `X`):

```
Fitur 2 ^
        |        X     X     X     X
        |     X     +-------+     X
        |    X      | O   O |      X   <--- Batas Melingkar Mulus
        |     X     |   O   |     X
        |        X  +-------+  X
        +----------------------------------->
                                       Fitur 1
```

#### D. Batas Berliku Ekstrem (Overfitting)
Model terlalu berusaha membenarkan semua titik data latihan sehingga batasnya meliuk-liuk aneh mengitari titik pencilan tunggal:

```
Fitur 2 ^
        |      O      O       X      X
        |   O      O     +--+     X
        |      O         |O |  X      X  <--- Melingkari 1 data pencilan
        |   O      O     +--+      X
        +----------------------------------->
                                       Fitur 1
```

---

### 4.3 Sumber Rujukan Topik 4
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 4: “Linear Models for Classification”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 3: “Classification”.

---

## 5. Model Linear vs Non-Linear (Secara Konseptual)

Perbedaan paling mendasar dalam arsitektur model adalah apakah model tersebut hanya mampu memetakan hubungan lurus atau sanggup memetakan pola yang melengkung dan berbelok.

---

### 5.1 Model Linear

#### A. Konsep Inti
Model linear beroperasi dengan asumsi bahwa target yang ingin diprediksi merupakan hasil **penjumlahan berbobot sederhana** dari fitur-fitur masukannya:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$

Secara geometris:
* Pada ruang dua dimensi (2 fitur), batasnya berupa **garis lurus**.
* Pada ruang tiga dimensi (3 fitur), batasnya berupa **bidang datar**.
* Pada dimensi yang lebih tinggi, batasnya berupa **bidang datar berdimensi banyak (*hyperplane*)**.

#### B. Kelebihan:
* **Sangat Cepat**: perhitungan matematika hanya berupa perkalian matriks sederhana sehingga sangat efisien saat melayani inferensi di peladen produksi.
* **Mudah Dipahami (*Interpretable*)**: kita bisa langsung menjelaskan alasan di balik keputusan model kepada regulator atau pemangku kepentingan. Jika bobot $w_1$ bernilai positif besar, artinya Fitur 1 berbanding lurus dan berpengaruh kuat terhadap target tebakan.
* **Tahan Banting**: tidak mudah mengalami kondisi menghafal mati (*low variance*), sangat stabil jika data latihnya berukuran kecil.

#### C. Kelemahan:
* **Kapasitas Terbatas**: gagal total jika data memiliki hubungan interaksi non-linear yang rumit (seperti masalah logika *Exclusive OR* / XOR).

#### D. Contoh Algoritma:
Linear Regression, Logistic Regression, Linear Support Vector Machines, dan Perceptron Sederhana.

---

### 5.2 Model Non-Linear

#### A. Konsep Inti
Model non-linear tidak membatasi diri pada perkalian skalar lurus. Model ini mampu memodelkan kurva berderajat banyak, permukaan bergelombang, pemotongan hierarki bertingkat, serta pola klaster yang terisolasi.

Model non-linear dapat menangkap pola di mana efek dari satu fitur bergantung pada kondisi fitur lainnya. Misalnya: obat dosis tinggi bermanfaat jika usia pasien dewasa, tetapi berbahaya jika pasien masih anak-anak (hubungan interaksi bersilang).

#### B. Kelebihan:
* **Fleksibilitas Tinggi**: sanggup mengekstrak struktur pola yang sangat abstrak pada data persepsi manusia seperti gambar foto, rekaman audio, dan teks bahasa alami.
* **Daya Prediksi Superior**: jika jumlah data tersedia melimpah, akurasi model non-linear sering kali melampaui model linear.

#### C. Kelemahan:
* **Rawan Menghafal Mati (*High Variance*)**: model mudah terjebak mempelajari derau atau kebetulan semata jika tidak dibatasi dengan teknik regularisasi yang baik.
* **Sulit Dipahami (*Black Box*)**: sangat sulit melacak mengapa jaringan saraf tiruan dengan puluhan lapisan mengambil kesimpulan tertentu dari jutaan parameter bobotnya.
* **Beban Komputasi Besar**: membutuhkan waktu pelatihan yang lama serta konsumsi memori dan daya komputasi (GPU) yang tinggi.

#### D. Contoh Algoritma:
Kernel SVM (Radial Basis Function / RBF), Decision Trees, Random Forest, Gradient Boosted Trees (XGBoost/LightGBM), k-Nearest Neighbors, dan Jaringan Saraf Tiruan (*Deep Neural Networks*).

---

### 5.3 Tabel Komparasi Model Linear vs Non-Linear
| Dimensi Evaluasi | Model Linear | Model Non-Linear |
| :--- | :--- | :--- |
| **Bentuk Batas Keputusan** | Selalu berupa garis lurus / bidang datar | Berupa kurva bebas, lingkaran, atau bidang bergelombang |
| **Kebutuhan Volume Data** | Berfungsi baik meski data berukuran kecil | Membutuhkan data dalam jumlah besar agar stabil |
| **Interpretabilitas** | Sangat tinggi (alur logika mudah diurai) | Cenderung rendah hingga sulit diurai (*black-box*) |
| **Beban Komputasi** | Sangat ringan dan hemat daya | Berat, membutuhkan proses komputasi intensif |
| **Risiko Masalah Utama** | Rentan kurang pas (*underfitting*) | Rentan menghafal mati (*overfitting*) |

---

### 5.4 Sumber Rujukan Topik 5
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 3: “Linear Methods for Regression” dan Bab 4: “Linear Methods for Classification”.
* **Breiman, Leo (2001)**, “Statistical Modeling: The Two Cultures”, *Statistical Science*, 16(3), hlm. 199–231.
* **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**, *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2, Springer, Bab 2: “Statistical Learning”.

---

## 6. Kerangka Rangkuman & Peta Konsep

Gunakan bagan alur di bawah ini untuk memahami hubungan berkesinambungan antara konsep model:

```
                            [ TUJUAN SISTEM ML ]
                                     |
                                     v
                       [ PILIH KELUARGA ALGORITMA ]
                                     |
                                     v
             +-----------------------------------------------+
             | MENETAPKAN RUANG HIPOTESIS                    |
             | - Menentukan bentuk batasan (Bias Induktif)   |
             | - Mengatur SETELAN LUAR (Hyperparameters)     |
             +-----------------------------------------------+
                                     |
                                     v
             +-----------------------------------------------+
             | PROSES PELATIHAN DARI DATA                    |
             | - Menggeser dan mencari nilai PARAMETER DALAM |
             |   (Bobot w dan Bias b) hingga galat mengecil  |
             +-----------------------------------------------+
                                     |
                                     v
             +-----------------------------------------------+
             | LAHIRLAH SEBUAH MODEL                         |
             | - Membentuk BATAS KEPUTUSAN                   |
             |   (Linear: garis lurus vs Non-Linear: kurva)  |
             +-----------------------------------------------+
                                     |
                                     v
                   [ UJI KEMAMPUAN GENERALISASI ]
                   (Apakah tebakannya akurat pada data baru?)
```

---

## 7. Daftar Pustaka Terverifikasi

1. **Mitchell, Tom M. (1997)**. *Machine Learning*. McGraw-Hill Education.
2. **Bishop, Christopher M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.
3. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
4. **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**. *Deep Learning*. MIT Press.
5. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
6. **Breiman, Leo (2001)**. “Statistical Modeling: The Two Cultures”. *Statistical Science*, 16(3), hlm. 199–231.
7. **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**. *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2. Springer.
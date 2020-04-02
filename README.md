# Delta

Megoldás leírása:  
Betöltjük a fájlok felét a mappából a tanítási halmazba, a másik felét pedig a tesztelési halmazba. A tanítási halmaz tartalmára meghívjuk a offline tanítási módszert, logsig lépcső függvénnyel. Következőben meghívjuk a predict függvényt a tesztelési halmazunkra és a neuronhálonkra, ami lefoglalja a szükséges méretet a kimenetre. Meghívjuk a tesztelési halmazunkra az osztályozó függvényt. Ezután kirajzoljuk grafikusan a tesztelési halmaz egy részét az osztályozó kimenetével és megjelenítsük a konfuziós hálót.

Problémák:
Hiányzó útmutató a feladat megoldására.  
Matlab 2016-os verziójában nem elérhető a konfuziós mátrix könyvtára.

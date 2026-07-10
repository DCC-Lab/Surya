puissance: 64.3 mW
on store le background à 100 ms d'intégration

je prends 10 acquisition pour 3 zones pour chaque échantillon à 20 000 ms.
- [x] P3S2.2J8
- [x] P4S4J8
- [x] P4S5J8
- [x] P5S4J8 très petit, je sais pas a quelle pt la zone 3 est différente de la zone 2
- [x] PS5J8<

# Traitement données RAMAN

donc la on doit d'abord individuellement pour chaque acquisition:
1. corriger la fluorescence et le rayonnement cosmique
2. moyenner les 10 acquisitions
3. diviser par le temps d'intégration
4. combiner (additionner les 3 zones)
j'ai réussi a faire quelque graphique, cependant, je trouve étrange le gros pic à 1362.0 cm⁻¹. Claude semble dire que c'est attendu, car ce pic est cohérent avec :

- **Liaisons C-C et C-N** des protéines et lipides
- **Collagène** et autres protéines structurales du tissu
- **Bandes D du carbone désordonné** présent dans les tissus biologiques

 au total, cela donne 10 spectres 

| petri         | souri   |
| ------------- | ------- |
| petri 1 0 gy  | 1, 2, 3 |
| petri 2 45 gy | 1, 2, 3 |
| petri 3 60 gy | 4, 5    |
| petri 4 80 gy | 4, 5    |

![[signal_raman_4_petri.png]]
souri 2 45 gy est assez laid...
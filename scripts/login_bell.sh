#!/bin/bash

# On initialise le nombre de personnes connectées actuellement
prev_count=$(who | wc -l)

echo "Surveillance des connexions active... (Ctrl+C pour quitter)"

while true; do
    # On revérifie toutes les 2 secondes
    sleep 2
    current_count=$(who | wc -l)

    # Si le nombre actuel est plus grand que le précédent, quelqu'un s'est connecté
    if [ "$current_count" -gt "$prev_count" ]; then
       
        echo "$(date '+%H:%M:%S') - Nouvelle connexion détectée !"

        # Boucle pour faire sonner la cloche 10 fois
	for i in $(seq 1 30); do
	    echo "Bip numero $i / 10"
            printf '\a'
            sleep 0.3 # Un mini délai pour que les "bips" ne se chevauchent pas
        done
    fi

    # On met à jour le compteur pour la prochaine vérification
    prev_count=$current_count
done

# Import dependencies #
import numpy as np
import googlemaps
import gmplot

from shared.constant import *

api_key = 'AIzaSyCUTdqn7zojehUSyRt0xZGk_kKk1-A9cKI'
gmaps = googlemaps.Client(key=api_key)

# Code starts here #


def generate_ants(num_ants, num_nodes, start_node):
    ants = np.zeros((num_ants, num_nodes), dtype=int)
    ants[:, 1:] = np.arange(num_ants)[:, np.newaxis]
    for ant in ants:
        np.random.shuffle(ant[1:])
    ants[:, 0] = start_node
    return ants


def move_ant(ant, pheromones, inv_distances, durations, alpha, beta):
    n = len(ant)
    visited = np.zeros(n, dtype=bool)
    visited[ant[0]] = True

    for i in range(1, n):
        not_visited = np.where(~visited)[0]
        probabilities = pheromones[ant[i-1], not_visited]**alpha * \
            inv_distances[ant[i-1], not_visited]**beta * \
            (1 / durations[ant[i-1], not_visited])

        if np.sum(probabilities) == 0:
            probabilities = np.ones_like(probabilities) / len(not_visited)
        else:
            probabilities /= np.sum(probabilities)

        if np.any(np.isnan(probabilities)):
            probabilities = np.ones_like(probabilities) / len(not_visited)

        ant[i] = np.random.choice(not_visited, p=probabilities)
        visited[ant[i]] = True

    return ant


def get_distance(path, distances):
    total_distance = 0.0
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        segment_distance = distances[node1, node2]
        total_distance += segment_distance
    return total_distance


def get_duration(path, durations):
    total_duration = 0.0
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        segment_duration = durations[node1, node2]
        total_duration += segment_duration
    return total_duration


def ant_colony(num_nodes, latitudes, longitudes, distances, durations, places, pheromone=0.5, alpha=1, beta=7, evaporation=0.5, ants=30, iterations=200, early_stopping_generations=10):
    num_ants = ants
    num_generations = 0
    best_path = np.arange(num_nodes)
    best_distance = get_distance(best_path, distances) 
    best_duration = get_duration(best_path, durations)
    max_distance = np.max(distances)
    early_stopping_rounds = 0
    same_distance_count = 0
    consecutive_same_distance_count = 0

    # Find the index of the start node in the places array
    start_node_index = 0

    while num_generations < iterations:
        num_generations += 1
        pheromones = np.ones_like(distances) * pheromone
        inv_distances = 1 / (distances + 1e-10)

        pheromones *= (1 - evaporation)
        ants = generate_ants(num_ants, num_nodes, start_node_index)
        ants = [move_ant(ant, pheromones, inv_distances, durations, alpha, beta) for ant in ants]

        for ant in ants:
            distance = get_distance(ant, distances)
            duration = get_duration(ant, durations)

            if distance < best_distance: 
                best_path = np.copy(ant)
                best_distance = distance
                best_duration = duration
                plot_solution(latitudes, longitudes, best_path, places, durations)
                same_distance_count = 0  # Reset the count when a better distance is found
            else:
                same_distance_count += 1

            Q = max_distance * num_nodes

            for i in range(num_nodes - 1):
                pheromones[ant[i], ant[i + 1]] += Q / distance
                pheromones[ant[i + 1], ant[i]] += Q / distance

        early_stopping_rounds += 1
        print('---------')
        print('Generation:', num_generations)
        print('Best Path:', ' - '.join([str(places[i]["id"]) for i in best_path]))
        print('Best Distance:', f'{best_distance:.2f} km')
        print('Best Duration:', f'{best_duration:.2f} minutes')

        if same_distance_count >= 10:
            consecutive_same_distance_count += 1
            if consecutive_same_distance_count >= early_stopping_generations:
                break

        if early_stopping_rounds >= num_nodes * early_stopping_generations:
            break

    print('---------')
    print('## Best Results ##')
    print('Total Generations:', num_generations)
    print('Best Path:', ' - '.join([str(places[i]["id"]) for i in best_path]))
    print('Best Distance:', f'{best_distance:.2f} km')
    print('Best Duration:', f'{best_duration:.2f} minutes')

    return best_path, best_distance
# Get Distances and


def calculate_distance(coord1, coord2):
    result = gmaps.directions(
        coord1, coord2, mode='driving', departure_time="now", traffic_model='best_guess')
    distance = result[0]['legs'][0]['distance']['value'] / 1000
    duration = result[0]['legs'][0]['duration']['value'] / 60
    return distance, duration


def calculate_distances_durations(places, latitudes, longitudes):
    num_nodes = len(places)
    distances = np.zeros((num_nodes, num_nodes))
    durations = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            coord1 = (latitudes[i], longitudes[i])
            coord2 = (latitudes[j], longitudes[j])
            distance, duration = calculate_distance(coord1, coord2)
            distances[i][j] = distance
            durations[i][j] = duration

    return distances, durations


def solve_and_plot(num_nodes, places, pheromone=0.5, alpha=1, beta=7, evaporation=1, ants=30, iterations=200, early_stopping_generations=10):
    latitudes = np.zeros(num_nodes)
    longitudes = np.zeros(num_nodes)

    for i in range(num_nodes):
        latitudes[i] = places[i]["lat"]
        longitudes[i] = places[i]["long"]

    distances, durations = calculate_distances_durations(
        places, latitudes, longitudes)

    best_path, best_distance = ant_colony(num_nodes, latitudes, longitudes, distances, durations,
                                          places, pheromone, alpha, beta, evaporation, ants, iterations, early_stopping_generations)
    best_duration = get_distance(best_path, durations)

    # Plot the solution
    plot_solution(latitudes, longitudes, best_path, places, durations)

    return best_path, best_distance, best_duration


def plot_solution(latitudes, longitudes, path, places, durations):
    path_coords = [(latitudes[node], longitudes[node]) for node in path]
    directions_result = gmaps.directions(
        path_coords[0], path_coords[-1], waypoints=path_coords[1:-1], optimize_waypoints=False, mode='driving')
    polyline = directions_result[0]['overview_polyline']['points']
    decoded_polyline = googlemaps.convert.decode_polyline(polyline)
    latitudes = [coord['lat'] for coord in decoded_polyline]
    longitudes = [coord['lng'] for coord in decoded_polyline]

    # Create a Google MapsPlotter object centered at the starting point (Politeknik Elektronika)
    zoom_level = 13
    gmap = gmplot.GoogleMapPlotter(
        path_coords[0][0], path_coords[0][1], zoom=zoom_level, apikey=api_key)

    # Create a polyline with the route coordinates (With lightblue color)
    gmap.plot(latitudes, longitudes, '#00b0ff', edge_width=4)

    # Add markers for the nodes with ID numbers and place names
    for i, (coord, place) in enumerate(zip(path_coords, places)):
        marker_label = chr(65 + i)
        gmap.marker(coord[0], coord[1],
                    title=place["name"], label=marker_label)

    # Add markers for the start and end points
    start_marker_label = "A"  # Start point labeled as A
    # End point labeled as the last alphabet
    end_marker_label = chr(64 + len(path))
    gmap.marker(path_coords[0][0], path_coords[0][1],
                title="Start", color='red', label=start_marker_label)
    gmap.marker(path_coords[-1][0], path_coords[-1][1],
                title="End", color='red', label=end_marker_label)

    # Calculate distance for each segment of the path using Google Maps actual distance
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        segment_distance = directions_result[0]['legs'][i]['distance']['value'] / 1000
        segment_duration = durations[node1, node2]
        info_text = f"Segment {chr(65 + i)} to {chr(65 + i + 1)}: Distance = {segment_distance:.2f} km, Time = {segment_duration}"
        gmap.text((latitudes[node1] + latitudes[node2]) / 2, (longitudes[node1] +
                  longitudes[node2]) / 2, info_text, color='black', fontsize=10)

    # Generate the HTML file to display the map
    gmap.draw('map.html')


# Initialize the code #
places = [
    {"id": 1, "name": "Politeknik Elektronika Negeri Surabaya", "lat": -7.27584, "long": 112.79118},
    # {"id": 1, "name": "Balai Kota Surabaya", "lat": -7.259131171716397, "long": 112.74710619999999},
    {"id": 2, "name": "Ekowisata Mangrove", "lat": -7.308585782197063, "long": 112.82139066843838},
    # {"id": 3, "name": "Pantai Kenjeran", "lat": -7.237891299293556, "long": 112.79554308179262},
    # {"id": 4, "name": "Hutan Bambu Keputih", "lat": -7.293903558328975, "long": 112.8017667106283 },
    # {"id": 5, "name": "Masjid Sunan Ampel", "lat": -7.230164996386202, "long": 112.74285225111173},
    # {"id": 6, "name": "Masjid Muhammad Cheng Hoo", "lat": -7.251283988632909, "long": 112.74676536753938},
    # {"id": 7, "name": "Masjid Agung Surabaya", "lat": -7.336273580563577, "long": 112.71518059528827},
    # {"id": 8, "name": "Museum Perjuangan 10 Nopember", "lat": -7.2445989517793175, "long":112.73777364295655},
    # {"id": 9, "name": "Monumen Kapal Selam", "lat": -7.265387821966468, "long": 112.75034811062804},
    # {"id": 10, "name": "Museum House of Sampoerna", "lat": -7.233733566542891,"long": 112.73104999985851}, 
    # {"id": 11, "name": "Taman Bungkul", "lat": -7.291112666929594, "long": 112.73982179528782},
    # {"id": 12, "name": "Kebun Binatang Surabaya", "lat":  -7.294410550822924, "long":  112.73664724943053}, 
    # {"id": 12, "name": "Pasar Atom", "lat": -7.241474141113638, "long": 112.74398968179264},
    # {"id": 14, "name": "Pasar Turi", "lat":  -7.245731513861289, "long": 112.73212946460661},
    # {"id": 15, "name": "Depot Bu Rudy", "lat": -7.267119982373765, "long": 112.77015073946305},
    # {"id": 16, "name": "Museum Pendidikan", "lat": -7.255699646366314, "long": 112.74280870113613}  
    # Add more places as needed
]

num_nodes = len(places)
solve_and_plot(num_nodes, places, pheromone=0.5, alpha=0.5, beta=7, evaporation=1, ants=30, iterations=200, early_stopping_generations=10)

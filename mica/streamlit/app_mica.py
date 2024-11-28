import streamlit as st
import pickle
import pandas as pd
import math
import os

# Definiciones de funciones necesarias para la deserialización
def pearson_similarity(person1, person2, data):
    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
    n = len(common_ranked_items)
    s1 = sum([data[person1][item] for item in common_ranked_items])
    s2 = sum([data[person2][item] for item in common_ranked_items])
    ss1 = sum([pow(data[person1][item], 2) for item in common_ranked_items])
    ss2 = sum([pow(data[person2][item], 2) for item in common_ranked_items])
    ps = sum([data[person1][item] * data[person2][item] for item in common_ranked_items])
    num = n * ps - (s1 * s2)
    den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))
    return (num / den) if den != 0 else 0

def normalize_rating(rating, min_rating=1, max_rating=5):
    min_possible_rating = 0
    max_possible_rating = 10
    normalized_rating = min_rating + (rating - min_possible_rating) * (max_rating - min_rating) / (max_possible_rating - min_possible_rating)
    return max(min_rating, min(max_rating, normalized_rating))

def recommend(person, bound, data, df_business):
    scores = [(pearson_similarity(person, other, data), other) for other in data if other != person]
    scores.sort(reverse=True, key=lambda x: x[0])
    scores = scores[:bound]

    recs = {}
    for sim, other in scores:
        ranked = data[other]
        for itm in ranked:
            if itm not in data[person]:
                weight = sim * ranked[itm]
                if itm in recs:
                    recs[itm] += weight
                else:
                    recs[itm] = weight

    recs_sorted = sorted(recs.items(), key=lambda x: x[1], reverse=True)
    filtered_business_ids = [b_id for b_id, _ in recs_sorted if b_id in df_business['business_id'].values]
    filtered_business_ids = filtered_business_ids[:bound]

    recommended_business_data = []
    for business_id in filtered_business_ids:
        business_info = df_business[df_business['business_id'] == business_id].iloc[0]
        rating = recs.get(business_id, 0)
        normalized_rating = normalize_rating(rating)

        recommended_business_data.append({
            'business_id': business_id,
            'name': business_info['name'],
            'address': business_info['address'],
            'city': business_info['city'],
            'rating': normalized_rating
        })

    recommended_business_info = pd.DataFrame(recommended_business_data)
    return recommended_business_info

# Cargar el modelo
model_path = os.path.join(os.path.dirname(__file__), '../Datos/recommend_model.pkl')
with open(model_path, 'rb') as file:
    data = pickle.load(file)

df_business = data['df_business']
data = data['data']

# Cargar el modelo y los datos
#with open('mica\Datos\recommend_model.pkl','rb') as file:
    #data = pickle.load(file)
    #df_business = data['df_business']  # Data de negocios

# Función de similitud de Pearson
def pearson_similarity(person1, person2, data):
    common_ranked_items = [itm for itm in data[person1] if itm in data[person2]]
    n = len(common_ranked_items)

    s1 = sum([data[person1][item] for item in common_ranked_items])
    s2 = sum([data[person2][item] for item in common_ranked_items])
    ss1 = sum([pow(data[person1][item], 2) for item in common_ranked_items])
    ss2 = sum([pow(data[person2][item], 2) for item in common_ranked_items])
    ps = sum([data[person1][item] * data[person2][item] for item in common_ranked_items])
    num = n * ps - (s1 * s2)
    den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))
    return (num / den) if den != 0 else 0

# Función de recomendación
def recommend(person, bound, data, df_business):
    # Calculamos la similitud con todos los demás usuarios
    scores = [(pearson_similarity(person, other, data), other) for other in data if other != person]
    
    # Ordenamos los puntajes en orden descendente
    scores.sort(reverse=True, key=lambda x: x[0])
    scores = scores[:bound]  # Limitar a la cantidad de recomendaciones deseadas

    # Crear un diccionario de negocios recomendados
    recs = {}
    for sim, other in scores:
        ranked = data[other]
        for itm in ranked:
            if itm not in data[person]:
                weight = sim * ranked[itm]
                if itm in recs:
                    recs[itm] += weight
                else:
                    recs[itm] = weight

    # Ordenar los negocios recomendados por el puntaje (de mayor a menor)
    recs_sorted = sorted(recs.items(), key=lambda x: x[1], reverse=True)

    # Filtrar los negocios que están en df_business
    filtered_business_ids = [b_id for b_id, _ in recs_sorted if b_id in df_business['business_id'].values]
    
    # Asegurarse de que el número de negocios recomendados no exceda el 'bound'
    filtered_business_ids = filtered_business_ids[:bound]

    # Crear una lista para almacenar los datos recomendados
    recommended_business_data = []

    for business_id in filtered_business_ids:
        business_info = df_business[df_business['business_id'] == business_id].iloc[0]
        rating = recs.get(business_id, 0)
        normalized_rating = normalize_rating(rating)  # Función que normaliza el rating

        recommended_business_data.append({
            'business_id': business_id,
            'name': business_info['name'],
            'address': business_info['address'],
            'city': business_info['city'],
            'rating': normalized_rating
        })

    # Convertir la lista en un DataFrame
    recommended_business_info = pd.DataFrame(recommended_business_data)
    return recommended_business_info

# Función para normalizar el rating
def normalize_rating(rating, min_rating=1, max_rating=5):
    min_possible_rating = 0
    max_possible_rating = 10
    normalized_rating = min_rating + (rating - min_possible_rating) * (max_rating - min_rating) / (max_possible_rating - min_possible_rating)
    return max(min_rating, min(max_rating, normalized_rating))

# Configuración de la página en Streamlit
st.set_page_config(page_title="Recomendación de Negocios", layout="wide")

# Título de la página
st.title("💼 Recomendación de Negocios")
st.markdown("Ingresa tu ID de usuario para obtener recomendaciones de negocios cercanos.")

# Entrada del ID de usuario y número de recomendaciones
user_id = st.text_input("Ingrese su ID de Usuario")
bound = st.slider("Número de recomendaciones", 1, 10, 5)

# Mostrar recomendaciones al hacer clic en el botón
if st.button("Obtener Recomendaciones"):

    if user_id in data:
        # Obtener las recomendaciones
        recomendaciones = recommend(user_id, bound, data, df_business)

        if recomendaciones.empty:
            st.warning("No se encontraron recomendaciones para este usuario.")
        else:
            st.success(f"Estos son los {len(recomendaciones)} negocios recomendados:")

            # Mostrar la tabla de recomendaciones
            st.dataframe(recomendaciones[['name', 'address', 'city', 'rating']])

    else:
        st.error("El ID de usuario ingresado no existe.")

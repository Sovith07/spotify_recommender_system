# set up the base image
FROM python:3.12

# set the working directory
WORKDIR /app/





# Copy all required data files at once
COPY ./data/preprocessed/collab_filtered_data.csv \
     ./data/preprocessed/sparse_matrix.npz \
     ./data/preprocessed/track_ids.npy \
     ./data/processed/cleaned_data.csv \
     ./data/preprocessed/transformed_data.npz \
     ./data/preprocessed/transformed_hybrid_data.npz \
     ./data/


# Copy all required Python scripts at once
COPY ./streamlit_app/app.py \
     ./src/collaborative_filtering.py \
     ./src/content_based_filtering.py \
     ./src/hybrid_recommendations.py \
     ./src/data_cleaning.py \
     ./src/transform_filtered_data.py \
     ./

# Copy the requirements file
COPY ./streamlit_app/dev_requirements.txt ./
     
# install the requirements
RUN pip install -r dev_requirements.txt
     
# expose the port on the container
EXPOSE 8000

# run the streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
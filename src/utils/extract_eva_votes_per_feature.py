import csv
import math

ATTRIBUTE = 'lightAndColor.csv'

input_file = 'datasets/eva/metadata/votes_filtered.csv'
output_file = 'datasets/eva/metadata/{}'.format(ATTRIBUTE)

if __name__ == "__main__":

    # Dictionary to store sum of scores and count of occurrences for each image_id
    image_scores = {}

    def sigmoid(x):
        return 10 * (1 / (1 + math.exp(-x)))

    with open(input_file, 'r') as f_in:
        reader = csv.reader(f_in, delimiter='=')
        
        # Skip the header row
        next(reader)
        
        # Iterate through each row in the input CSV file
        for row in reader:
            image_id = row[0]
            score = float(row[5])  # Convert score to float
            
            # If image_id is already in the dictionary, update the sum and count
            if image_id in image_scores:
                image_scores[image_id]['sum'] += score
                image_scores[image_id]['count'] += 1
            # Otherwise, initialize the sum and count for the image_id
            else:
                image_scores[image_id] = {'sum': score, 'count': 1}

    # Write the calculated average scores to the output CSV file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        
        # Write the header row
        writer.writerow(['image_path', 'average_score', 'transformed_score'])
        
        # Iterate through each image_id in the dictionary
        for image_id, data in image_scores.items():
            # Calculate the average score
            average_score = data['sum'] / data['count']

            image_path = 'datasets/eva/images/{}.jpg'.format(image_id)
            
            # Apply sigmoid transformation to the average score
            transformed_score = sigmoid(average_score - 2.5)  # Center the sigmoid around 2.5
            
            # Write the image_id and transformed_score to the output CSV file
            writer.writerow([image_path, average_score, transformed_score])
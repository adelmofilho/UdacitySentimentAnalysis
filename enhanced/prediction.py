import glob
import os
from tqdm import tqdm
import boto3

def predict(data, endpoint, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in tqdm(split_array):
        predictions = np.append(predictions, endpoint.predict(array))
    
    return predictions


def test_reviews(endpoint_name, data_dir='data/aclImdb', stop=5000):
    
    client = boto3.client("sagemaker-runtime")
    results = []
    ground = []
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        files_read = 0
        print('Starting ', sentiment, ' files')
        # Iterate through the files and send them to the predictor
        for f in tqdm(files):
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                try:
                    # Read in the review and convert to 'utf-8' for transmission via HTTP
                    review_input = review.read().encode('utf-8')
                    # Send the review to the predictor and store the results
                    response = client.invoke_endpoint(EndpointName=endpoint.endpoint, Body=review_input,ContentType='text/plain')
                    results.append(float(response["Body"].read().decode('utf-8')))
                    if sentiment == 'pos':
                        ground.append(1)
                    else:
                        ground.append(0)
                except:
                    pass
                #print(results)
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results
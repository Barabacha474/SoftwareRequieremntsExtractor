import csv
import random

# Define a set of long, meaningless descriptions for the "plain" descriptions in English
long_plain_templates_en = [
    "In light of our ongoing efforts to enhance user interaction with our web platform, we must consider implementing functionality that allows users to leave their feedback and impressions about the products and services available on our website. This not only increases customer engagement but also fosters an environment of trust and openness where every voice matters.",

    "To improve overall efficiency and optimize workflow within our team, we have realized that automating the reporting processes is one of the key directions we need to pursue. Currently, reporting requires significant time and effort, leading to ineffective resource allocation.",

    "In the context of our continuous pursuit of development and the need to keep our audience informed about the latest events, we must create a dedicated news page on our website, serving as a centralized source of information where users can find updates about our events and innovations.",

    "Given the importance of data security and the need to protect it from potential losses due to technical issues, we need to develop a backup system that ensures data availability at all times.",

    "In response to user complaints about the application loading with noticeable delays, we need to address this issue and take steps aimed at improving performance.",

    "To ensure effective operation within our team, it is crucial to implement a system that automates processes related to project and task management, which will undoubtedly increase overall productivity.",

    "Considering the growing security demands, we must look into integrating modern data protection methods into our existing systems, which will significantly enhance user trust.",

    "In light of current trends, it is important to create an adaptive interface for our application that can easily adjust to various devices, providing a better user experience."
]

# Define corresponding technical descriptions with variations in English
technical_descriptions_en = [
    "Implement a feedback system on the website, allowing users to share their opinions on products and services.",
    "Optimize the application for better performance by analyzing loading times and implementing caching.",
    "Automate the report generation process to ensure regular data collection without manual intervention.",
    "Create a news page on the website with functionalities for adding, editing, and deleting entries.",
    "Set up daily backups for the database with automated processes and failure notifications.",
    "Develop a task management system, including status tracking and deadlines.",
    "Integrate a payment processing API while ensuring client data security and transaction safety.",
    "Design an adaptive interface for the mobile version of the site, optimizing it for different screen resolutions."
]


# Function to ensure adequate context in descriptions
def ensure_context(description, min_length=100):
    while len(description.split()) < min_length:
        description += " Additional details and information may be included to improve the understanding of the context of this task."
    return description


# Generate pairs of long plain and corresponding technical descriptions with randomization
long_task_descriptions_en = []

for _ in range(1000):  # Generate 100 random descriptions
    plain_description = random.choice(long_plain_templates_en)
    technical_description = random.choice(technical_descriptions_en)

    # Ensure the plain description has enough context
    plain_description = ensure_context(plain_description)

    long_task_descriptions_en.append([_, plain_description, technical_description])

# Define file path for the new CSV file
long_file_path_en = 'long_task_descriptions_en.csv'

# Write the long task descriptions to a CSV file
with open(long_file_path_en, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "fulltext", "essence"])
    writer.writerows(long_task_descriptions_en)

print(f"File '{long_file_path_en}' has been successfully created.")

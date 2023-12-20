def count_characters(input_string):
    # Use the len() function to get the length of the string
    num_characters = len(input_string)
    
    # Print the result
    print(f"The number of characters in the string is: {num_characters}")

# Get input from the user
user_input = input("Enter a string: ")

# Call the function to count characters
count_characters(user_input)

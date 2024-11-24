def remove_overlap_and_concatenate(text1, text2):
    """
    Concatenate two texts by removing the overlapping/repeated part at the boundary.
    
    Args:
    text1 (str): The first text portion (DB2 code from first image).
    text2 (str): The second text portion (DB2 code from second image).
    
    Returns:
    str: The concatenated text without overlap.
    """
    # Find the maximum overlap length between the end of text1 and the beginning of text2
    max_overlap = 0
    min_length = min(len(text1), len(text2))
    
    # Iterate over the possible overlap lengths
    for i in range(1, min_length + 1):
        if text1[-i:] == text2[:i]:
            max_overlap = i
    
    # Concatenate the text1 and text2 without the overlapping part
    concatenated_text = text1 + text2[max_overlap:]
    
    return concatenated_text

# Example usage
text_part1 = """
CREATE PROCEDURE example_proc AS BEGIN 
  SELECT * FROM table1; END
  
--tst comment1

--tst comment2

--tst comment3

--tst comment4

"""
text_part2 = """
--tst comment3

--tst comment4

  SELECT * FROM table2; END

"""

# Call the function to concatenate the text
concatenated_text = remove_overlap_and_concatenate(text_part1, text_part2)

print("Concatenated DB2 Stored Procedure:")
print(concatenated_text)

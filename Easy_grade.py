''' SCHOLAR_NETWORK_AUTOMATED_GRADE_ASSISTANCE
'''
#Importing Libraries
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from glob import glob
import pandas as pd

#VARIABLE DEFINITION
drop_list2, df1_col_rename, df2_col_rename, drop_list, std_col_rename, similarity_list, lt1, lt2 = [], [], [], [], [], [], [], []
word_lemma, word_stemmer, reg_words = WordNetLemmatizer(), PorterStemmer(), stopwords.words('english')
vector_sum = 0
ans, solution, marks = [], [], []

#HOMEPAGE
print("Scholar_network_automated_grade_assistance".upper().center(60, " "))
print()

print("Welcome to Easy Grade.",
          "We Automate Grading to save time.", sep = "\n")
          
print()

#MAIN CODE
def cleaning_processing_file1(file1):
    '''
    This function loads the Suggested answer csv file and process it for Analysis
    
    '''
    global df_solution, marks_dataframe, len_df_solution
    
    loading_questions = [col for col in file1.columns]
    marks_column = [q for q in loading_questions if q.startswith('Point')]
    
    #this groups the points columns into data frame
    marks_dataframe = file1[marks_column]
    len_df_mark = len(marks_dataframe.columns)
    
    for i in range(1, len_df_mark + 1):
        df1_col_rename.append(f'Q{i}_point')
        
    #Renaming the marks columns
    marks_dataframe.columns = df1_col_rename
    
    for col in marks_dataframe.columns:
        if col.startswith('Q') and marks_dataframe[col].dtypes == 'float64':
            marks_dataframe[col] = marks_dataframe[col].astype(int)
        else:
            pass
        
    #this removes Points columns
    for col in marks_column:
        if col in loading_questions:
            loading_questions.remove(col)
            
    # df_solution is the data frame containing the Suggested answer for all the questions
    df_solution = file1[loading_questions]
    len_df_solution = len(df_solution.columns)
    
    #this append the new default column names to df2_col_rename.
    for i in range(1, len_df_solution + 1):
        df2_col_rename.append(f'Sol_{i}')
        
    #renaming teacher's solution columns
    df_solution.columns = df2_col_rename


def cleaning_processing_file2(file2):
    '''
    This function loads the scripts csv file and process it for Analysis
    
    '''
    global sub_df
    
    loading2_questions = [i for i in file2 if not i.startswith("Points") and i != "Total points" and i != "Full Name" and i != "Participant Identification Number"]
    
    #std_df is students dataframe
    student_df = file2[loading2_questions]
    len_std = len(student_df.columns)
    
    #this append the new default column names to df_rename_col
    for i in range(1, len_std + 1):
        std_col_rename.append(f'Ans_{i}')
    
    #this renames the columns
    student_df.columns = std_col_rename
    
    #this concatenate the different columns sections 'the ID number and Name, the Question columns
    sub_df = pd.concat([file2[['Participant Identification Number', 'Full Name']], student_df], axis = 1, sort = False)

    #this strips leading and trailing whitespaces
    for col in sub_df.columns:
        if col.startswith('Ans') and sub_df[col].dtypes == 'object':
            sub_df[col] = sub_df[col].str.strip().str.lower()
        else:
            pass


def merging_dataframe():
    '''
    This function merges the two data frame and fills any empty or Nan values with
    a default value
    '''
    global df
    
    #this merges sub_df, df_solution, df_mark data frame
    df = pd.concat([sub_df, df_solution, marks_dataframe], axis = 1)
    
    #this update all the rows for the solution columns to aid comparison
    for i in range(0, len(df.index)):
        for item in df_solution.columns:
            df.at[i, item] = df.loc[0, item]
    
    #this update all the rows for the solution columns to aid comparison
    for i in range(0, len(df.index)):
        for item in marks_dataframe.columns:
            df.at[i, item] = df.loc[0, item]

def marking_answers():
    '''This function carries out a semantic analysis on the relationship between
    the suggested answer provided by the Teacher and Students answer.
    
    It also creates a Total column summing up the marks apportioned.
    
    '''
    
    #this maps the students answer and suggested solution into a nested ans list and a solution list.
    global df, drop_list2
    
    for i in range(len_df_solution):
        ans.append([])
        for val in df[f'Ans_{i +1}']:
            ans[i].append(val)
        for item in df[f'Sol_{i + 1}']:
            solution.append(item)
            break
    
    #this checks for similarities between each list values
    for i in range(len_df_solution):
        marks.append([])
        for j in range(len(df.index)):
             marks[i].append(text_similarity_model(ans[i][j], solution[i]))
    
    
    #Marks allocation
    for i in range(1, len_df_solution + 1):
        similarity_list = []
        #this allocates marks based on their percentage similarity
        df[f'Mark{i}'] = marks[i - 1]
        df.loc[df[f'Mark{i}'] > 0.25, f'Score{i}'] = df.loc[0, f'Q{i}_point']
        
        df.loc[df[f'Mark{i}'] < 0.25, f'Score{i}'] = df.loc[0, f'Q{i}_point'] - 1
        df.loc[df[f'Mark{i}'] < 0.10, f'Score{i}'] = df.loc[0, f'Q{i}_point'] - df.loc[0, f'Q{i}_point']
        
        #this creates a list of columns to be removed
        drop_list.append(f'Ans_{i}')
        drop_list.append(f'Sol_{i}')
        drop_list.append(f'Q{i}_point')
        drop_list.append(f'Mark{i}')
        
    #creating a new data frame excluding the dropped columns
    df = df.drop(drop_list, axis = 1)
    
    for i in df.columns:
        df[i] = df[i].fillna(0)
        
    df.loc[:,'Total_Score'] = df.sum(numeric_only=True, axis=1)
    drop_list2 = []
    
    for i in range(1, len_df_solution +1):
        drop_list2.append(f'Score{i}')
    
    
def getAndReplaceSynonymousWords(list1, list2):
	'''This function converts all words that are synonymous to the teachers word
	
	'''
	global student_ans
	
	for i in range(len(list1)):
		for synset in wordnet.synsets(list1[i]):
			for lemma in synset.lemma_names():
				for item in list2:
					if lemma == item:
						list2.remove(item)
						list2.append(list1[i])
	student_ans = list2
	return student_ans

def text_similarity_model(str1, str2):
    '''This function accepts two coloumns and calculates their similarities
    '''
    
    global reg_words, ans_similarity, vector_sum, output_df, summary_df
 
    lt1, lt2 = [], []
    vector_sum = 0
    col1_val = str(str1).lower()
    col2_val = str(str2).lower()
    
    #this creates a token of each elements (word) in a sentence in each columns
    col1_list = word_tokenize(col1_val)
    col2_list = word_tokenize(col2_val)
    
    #this removes regular words in stopwords like 'this', 'is', 'a', etc from the lists
    col1_set = {item for item in col1_list if item not in reg_words}
    col2_set = {item for item in col2_list if item not in reg_words}
    
    #this converts all words to their root word
    col1_root = [word_lemma.lemmatize(val) for val in col1_set]
    col2_root = [word_lemma.lemmatize(val) for val in col2_set]
    
    #Because of the limitation of WordNetLemmatizer; It doesn't converts plural words that ends with 'ed' to their root word ,e.g Loved = loved (expected 'love'). PorterStemmer handles the limitation of WordNetLemmatizer and vise-viser
    col1_revise_set = set()
    col2_revise_set = set()
    
    #To avoid the limitation of PortStemmer for e.g movie = movi (expected 'movie')
    for val in col1_root:
        if val.endswith('ed'):
            col1_root.remove(val)
            for i in col1_root:
                col1_revise_set.add(i)
                col1_revise_set.add(word_stemmer.stem(val))
        else:
            pass
        
    #To avoid the limitation of PortStemmer for e.g movie = movi (expected 'movie')
    for val in col2_root:
        if val.endswith('ed') == True:
            col2_root.remove(val)
            for i in col2_root:
                col2_revise_set.add(i)
                col2_revise_set.add(word_stemmer.stem(val))
        else:
            pass
    
    #if revise_set is empty then it will be filled with the root words previously gotten
    if col1_revise_set == set():
        for i in col1_root:
            col1_revise_set.add(i)
		
    if col2_revise_set == set():
        for i in col2_root:
            col2_revise_set.add(i)
    
    getAndReplaceSynonymousWords(list(col1_revise_set), list(col2_revise_set))
    rvector = col1_revise_set.union(student_ans)  
    
    for w in rvector:
        if w in col1_revise_set:
            lt1.append(1) # create a vector 
        else:
            lt1.append(0)
        if w in student_ans:
            lt2.append(1) 
        else:
            lt2.append(0) 
        
    # cosine formula  
    for i in range(len(rvector)): 
        vector_sum += lt1[i]*lt2[i] 
    
    try:
        ans_similarity = vector_sum / float((sum(lt1)*sum(lt2))**0.5)
    except ZeroDivisionError:
        ans_similarity = 0
    
    return ans_similarity

#GENERATING TWO CSV FILES
def exporting_two_files():
    '''
    This function exports two files.
    One containing a summary of the Students Performance and
    the other the Total Score of the students performance.
    
    '''
    global summary_df, output_df, drop_list2, org_name
    summary_df = df
    summary_df['Total_Score'] = summary_df['Total_Score'].astype(int)
    
    output_df = summary_df.drop(drop_list2, axis = 1)
    
    #This exports a new file containing the result
    #summary_df.to_csv(f'{org_name}ResultOverview.csv')
    
    #output_df.to_csv(f'{org_name}.csv')
    
    print("--" * 30)
    
    print(f"{org_name} Participant's Result >>>")
    print()
    print(output_df)
	
def main_execution():
	
    #This extracts the suggested answer and the students scripts
    global org_name
    
    file_path = [path for path in glob("*.csv")]
    if len(file_path) > 2:
    	print()
    	
    	print("Invalid File Location!!!",
    	          "Save this script and the two csv file in a new Directory.",
    	          "The Directory should only contain the script and the two csv files.", sep = '\n')
    	exit()
    else:
    	pass
    	          
    for path in file_path:
    	if path.lower() == "student_answers.csv":
    		students_csv = pd.read_csv(path)
    	elif path.lower() == "teachers_answer.csv":
    		teachers_csv = pd.read_csv(path)
    	else:
    		print()
    		
    		print("Invalid file name",
    		          "Ensure that the file Directory contains this script and the two csv file, named as 'script.csv' and 'suggested_answer.csv'", sep = "\n")
    		exit()
    
    org_name = input("Enter Organisation Name: ").upper()
    
    while org_name.isdigit() == True or len(org_name) < 3:
    	print()
    	
    	print("Unrecognised Organisational Name.")
    	print()
    	
    	org_name = input("Enter Organisation Name: ").upper()
    	
    print()
    print("Files are processing in 30 seconds.... ...... ....... ....")
    print()
    
    cleaning_processing_file1(file1 = teachers_csv)
    cleaning_processing_file2(file2 = students_csv)
    
    merging_dataframe()
    marking_answers()
    
    exporting_two_files()
    
#MAIN PROGRAM INVOCATION
main_execution()
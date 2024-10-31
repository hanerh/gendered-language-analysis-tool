import spacy
from spacy.tokens import Span, Token, Doc
from spacy.matcher import Matcher
from spacy import displacy
import gender_spacy.GenderSpacy as gs

import pandas as pd
from typing import Union
from pathlib import Path
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Define label constants for gendered terms
masc_label, fem_label, neutral_label = "MASC_TERM", "FEM_TERM", "NEUTRAL_TERM" 

# Define key for span list labels
label_key = ['glat', 'glat_term_matches', 'glat_excess_pronouns', 'glat_appropriate']

# CSV file containing gendered terms and their neutral counterparts
gendered_terms = "match_terms.csv"
df_match_terms = pd.read_csv(gendered_terms)


class GenderedLanguageAnalysisTool:
    """
    A tool to analyze spaCy documents for gendered language patterns and identify appropriate and unnecessary pronoun usage.
    Utilizes gendered term matching, pronoun tracking, and offers visualization options for identified patterns.
    
    Attributes:
        _doc (spacy.tokens.Doc): The spaCy Doc object after processing text.
        _appropriate_pronouns (dict): Dict tracking appropriate pronoun usage linked to gendered spans.
        _unnecessary_pronouns (dict): Dict tracking unnecessary (default) pronoun usage for each gender.
        _gendered_term_matches (list): List of spans matching gendered terms in text.
        _neutral_term_matches (list): List of spans matching neutral terms in text.
        _num_excess_pron_male (int): Number of unnecessary (default) masculine pronoun instances.
        _num_excess_pron_female (int): Number of unnecessary (default) feminine pronoun instances.
        _num_masc_terms (int): Count of identified masculine terms.
        _num_fem_terms (int): Count of identified feminine terms.
        _num_neutral_terms (int): Count of identified neutral terms.
    """
    
    def __init__(self) -> None:
        """
        Initializes a GenderedLanguageAnalysisTool object with variables to store analysis results.
        """
        # Initialize variables for storing analysis results
        self._doc = None
        self._appropriate_pronouns = {}
        self._unnecessary_pronouns = {"male": [], "female": []}
        self._gendered_term_matches = []
        self._neutral_term_matches = []
        self._num_excess_pron_male = 0
        self._num_excess_pron_female = 0
        self._num_masc_terms = 0
        self._num_fem_terms = 0
        self._num_neutral_terms = 0
        
        
    def _run_spacy(self, text: str) -> None:
        """
        Processes the input text with the spaCy GenderSpacy model, creating a Doc object with coreference resolution applied.

        Args:
            text (str): The text to analyze for gendered terms and pronouns.

        Raises:
            ValueError: If an empty string is provided for processing.
        """
        # Throw error if no text was given to analyze
        if text == "":
            raise ValueError("Can not process empty string")
        
        else:
            # Create the modified Gender-spaCy model
            self._nlp = gs.GenderSpacy()
            
            # Process the text and run coreference resolution
            self._nlp.process_doc(text)
            self._doc = self._nlp.coref_resolution() 

        
    def _tool_setup(self) -> None:
        """
        Configures the tool by applying a matcher for gendered terms to the Doc object and analyzing gendered pronoun use within the Doc.
        """
        self._find_matches()
        self._analyze_doc()     # Determines where gendered pronouns were used appropriately (for named entities ie. proper nouns) or unnecessarily (using gendered term as default)
        
        
    def analyse_text(self, text: str) -> None:
        """
        Analyzes the input text using the GenderSpacy model, identifying gendered language patterns.

        Args:
            text (str): The text to be analyzed.
        """
        # Process text with spaCy; assigning gender to text entities
        self._run_spacy(text)
        # Finish setting up the tool; run rule-based matcher to find gendered terms
        self._tool_setup()
        
        
    def save_tool(self, save_path: Union[str, Path]) -> None:
        """
        Saves the processed Doc to disk for future use.

        Args:
            save_path (Union[str, Path]): Path where the Doc should be saved.
        """
        self._doc.to_disk(save_path)
        
    
    def load_tool(self, path_to_saved) -> None:
        """
        Loads a saved Doc from disk and sets up the tool for further analysis.

        Args:
            path_to_saved (Union[str, Path]): Path to the saved Doc.
        """
        nlp = gs.GenderSpacy()
        self._doc = Doc(vocab=nlp.nlp.vocab).from_disk(path_to_saved)
        self._tool_setup()
        
            
    def _analyze_doc(self) -> None:
        """
        Analyzes the Doc to identify where gendered pronouns were used appropriately or unnecessarily.
        Tracks the usage of masculine, feminine, and neutral pronouns based on linked spans.
        Stores the results in internal attributes for later retrieval.
        """
        # Define pronouns to track for each gender
        pronoun_key = {
            "female": ["she", "her", "hers", "herself"], 
            "male": ["he", "him", "his", "himself"], 
            "neutral": ["they", "them", "their", "theirs", "themself"]
            }
        
        unnecessary_pronouns = {"male": {}, "female": {}}
        appropriate_pronouns = {}
        
        track_token_i = []  # To ensure pronouns are added only once
        group_counter = 0   # counter for group labels of spans
        # Iterate through gender spans in the Doc and classify pronouns used as appropriate or unnecessary
        for span in self._doc.spans["ruler"]:
            if span[0].pos_ in ["NOUN", "PROPN"]:
                
                span_pronouns = span._.gt_linked_pronouns
            
                ind_list = [token for token in span_pronouns]
                index_intersection = set(track_token_i).intersection(set(ind_list))
                
                # Helper function to filter out already processed tokens
                def mixed_tokens(span_prons, token_tracker):
                    ind_tokens = {}
                    for ind, text in span_prons.items():
                            if ind not in token_tracker:
                                ind_tokens.update({ind: text})
                    return ind_tokens
                                
                if len(index_intersection) != 0:
                    span_pronouns = mixed_tokens(span_pronouns, track_token_i)
            
                # Perform case-insensitive intersections
                masc_intersection = set(pronoun.lower() for pronoun in pronoun_key["male"]).intersection(set(pronoun.lower() for pronoun in list(span_pronouns.values())))
                fem_intersection = set(pronoun.lower() for pronoun in pronoun_key["female"]).intersection(set(pronoun.lower() for pronoun in list(span_pronouns.values())))
                neutral_intersection = set(pronoun.lower() for pronoun in pronoun_key["neutral"]).intersection(set(pronoun.lower() for pronoun in list(span_pronouns.values())))

                # Helper function to create pronoun spans from the indices
                def create_pronoun_spans(pronouns, gender):
                    new_spans = []
                    for i in pronouns.keys():
                        new_spans.append(Span(self._doc, i, i+1, f"{gender.upper()}_PRONOUN"))
                    return new_spans
                
                # Helper function to distinguish connected tokens in spans from the other spans via numbers in their labels
                def grouplabel_spans(head_span, pronoun_spans, group_num):
                    span_label = f"{head_span.label_}_N{group_num}"
                    head_span.label_ = span_label
                    
                    for pron in pronoun_spans:
                        pron_label = f"{pron.label_}_{group_num}"
                        pron.label_ = pron_label
                    
                    return head_span, pronoun_spans
                
                # Analyze whether the span is unnecessary or appropriate based on part-of-speech tags
                if (span[0].pos_ == "NOUN"):
                    if len(masc_intersection) != 0:  
                        gender = 'male' 
                        pronouns = create_pronoun_spans(span_pronouns, gender)
                        span, pronouns = grouplabel_spans(span, pronouns, group_counter)
                        unnecessary_pronouns[gender].update({span:pronouns})
                        
                        group_counter += 1
                    elif len(fem_intersection) != 0:
                        gender = 'female'
                        pronouns = create_pronoun_spans(span_pronouns, gender)
                        span, pronouns = grouplabel_spans(span, pronouns, group_counter)
                        unnecessary_pronouns[gender].update({span:pronouns})
                        group_counter += 1
                        
                    elif len(neutral_intersection) != 0:
                        gender = 'neutral'
                        pronouns = create_pronoun_spans(span_pronouns, gender)
                        span, pronouns = grouplabel_spans(span, pronouns, group_counter)
                        appropriate_pronouns.update({span:pronouns}) 
                        group_counter += 1
                
                    else:
                        continue
                    
                elif (span[0].pos_ == "PROPN") and (len(masc_intersection) or len(fem_intersection) or len(neutral_intersection)):
                    gender = 'male' if len(masc_intersection) else 'female' if len(fem_intersection) else 'neutral' if len(neutral_intersection) else ""
                    pronouns = create_pronoun_spans(span_pronouns, gender)
                    span, pronouns = grouplabel_spans(span, pronouns, group_counter)
                    appropriate_pronouns.update({span:pronouns})
                    group_counter += 1

                for ind in span_pronouns:
                    track_token_i.append(ind)
            
        # Set corresponding class attributes
        self._appropriate_pronouns = appropriate_pronouns
        self._unnecessary_pronouns = unnecessary_pronouns
        self._num_excess_pron_male = len(unnecessary_pronouns["male"])
        self._num_excess_pron_female = len(unnecessary_pronouns["female"])    
        
        # Store results in the Doc's spans 

        if self._doc.spans.get('glat_appropriate', None) == None:
            self._doc.spans['glat_appropriate'] = []
            for noun, list_pron in appropriate_pronouns.items():
                    self._doc.spans["glat_appropriate"].append(noun)
                    for pron in list_pron:
                        self._doc.spans['glat_appropriate'].append(pron)  
            
        if self._doc.spans.get('glat_excess_pronouns', None) == None: 
            self._doc.spans['glat_excess_pronouns'] = []       
            for gender in unnecessary_pronouns.keys():
                for noun, list_pron in unnecessary_pronouns[gender].items():
                    self._doc.spans["glat_excess_pronouns"].append(noun)
                    for pron in list_pron:
                        self._doc.spans['glat_excess_pronouns'].append(pron)      
            self._doc.spans['glat'] = [] if (self._doc.spans.get("glat", None) == None) else self._doc.spans['glat']
            for span in self._doc.spans['glat_excess_pronouns']:
                self._doc.spans['glat'].append(span)
        

    def _create_gendered_matcher(self) -> Matcher:
        """
        Creates and configures a spaCy Matcher to identify gendered terms in the text based on CSV data.

        Returns:
            Matcher: Configured Matcher object for gendered terms.
        """
        # create the spacy Matcher instance
        matcher = Matcher(self._doc.vocab)
        
        def add_gendered_patterns_from_csv(df_terms: pd.DataFrame, term_matcher: Matcher) -> None:
            """
            Adds patterns from the CSV DataFrame to the spaCy Matcher.
            Handles multi-word terms and adds POS constraints to match only nouns.

            Args:
                df_terms (pd.DataFrame): DataFrame containing gendered terms and their neutral counterparts.
                term_matcher (spacy.matcher.Matcher): The spaCy matcher to add the patterns to.
            """
            masc_patterns, fem_patterns, neutral_patterns = [], [], []
            for _, row in df_terms.iterrows():
                # Build patterns for male, female, and neutral terms
                male_term = row.get("male_term", None)
                male_plural = row.get("male_plural", None)
                neutral_term = row.get("neutral_term", None)
                neutral_plural = row.get("neutral_plural", None)
                female_term = row.get("female_term", None)
                female_plural = row.get("female_plural", None)

                if pd.notna(male_term):
                    male_term_tokens = male_term.split()
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in male_term_tokens]
                    masc_patterns.append(pattern)
                if pd.notna(male_plural):
                    male_plural_tokens = male_plural.split()
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in male_plural_tokens]
                    masc_patterns.append(pattern) 
                if pd.notna(female_term):
                    female_term_tokens = female_term.split()
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in female_term_tokens]
                    fem_patterns.append(pattern)
                if pd.notna(female_plural):
                    female_plural_tokens = female_plural.split()
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in female_plural_tokens]
                    fem_patterns.append(pattern)
                if pd.notna(neutral_term):
                    neutral_term_tokens = [nt.strip() for nt in row.get("neutral_term", "").split(",")] if pd.notna(row['neutral_term']) else []
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in neutral_term_tokens]
                    neutral_patterns.append(pattern)
                if pd.notna(neutral_plural):
                    neutral_plural_tokens = [np.strip() for np in row.get("neutral_plural", "").split(",")] if pd.notna(row['neutral_plural']) else []
                    pattern = [{"LOWER": token.lower(), "POS": "NOUN"} for token in neutral_plural_tokens]
                    neutral_patterns.append(pattern)
            
            # Add patterns to the matcher
            term_matcher.add(masc_label, masc_patterns)
            term_matcher.add(fem_label, fem_patterns)
            term_matcher.add(neutral_label, neutral_patterns)
        
        # Call function to add matcher patterns from the CSV
        add_gendered_patterns_from_csv(df_match_terms, matcher)
        
        return matcher
    
    
    def _find_matches(self) -> None:
        """
        Finds and labels gendered term matches in the Doc, storing identified terms in class attributes.
        """
        # Instantiate a matcher for gendered terms
        matcher = self._create_gendered_matcher()
        
        # Create a dictionary to map gendered terms (male/female) to their neutral terms
        term_mapping = {}

        for _, row in df_match_terms.iterrows():
            # Map male singular and plural terms to their corresponding neutral terms
            if pd.notna(row['male_term']) and pd.notna(row['neutral_term']):
                term_mapping[row['male_term']] = row['neutral_term']
            if pd.notna(row['male_plural']) and pd.notna(row['neutral_plural']):
                term_mapping[row['male_plural']] = row['neutral_plural']
            
            # Map female singular and plural terms to their corresponding neutral terms
            if pd.notna(row['female_term']) and pd.notna(row['neutral_term']):
                term_mapping[row['female_term']] = row['neutral_term']
            if pd.notna(row['female_plural']) and pd.notna(row['neutral_plural']):
                term_mapping[row['female_plural']] = row['neutral_plural']
        
        def assign_neutral_label(match: Span, mapping: dict) -> None:
            """
            Assigns the corresponding neutral term as a label to a matched gendered term if found in the mapping dict.

            Args:
                match (spacy.tokens.Span): The matched span to relabel.
                mapping (dict): Dictionary to map gendered terms (male/female) to their neutral terms.
            """
            token_text = match.text.lower()  # Get the matched token text
            
            # Check if the term needs to be relabeled using the term_mapping
            if token_text in mapping:
                # Get the corresponding neutral term from the CSV
                neutral_term = mapping[token_text].capitalize()
                # Relabel the match span with the neutral term
                match.label_ = neutral_term
                
            
        # Run the matcher on the Doc and gather matches
        matches = matcher(self._doc, as_spans=True)
        
        gendered_term_matches, neutral_term_matches, all_term_matches = [], [], []
        relevant_spans = list(self._doc.spans["ruler"])
        
        num_masc_terms, num_fem_terms, num_neutral_terms = 0, 0, 0
        for match in matches:
            # Process each match and categorize it based on its label
            if match.label_ == masc_label:
                num_masc_terms += 1
                gendered_term_matches.append(match)
            elif match.label_ == fem_label:
                num_fem_terms += 1
                gendered_term_matches.append(match)
            elif match.label_ == neutral_label:
                num_neutral_terms += 1
                neutral_term_matches.append(match)
            else:
                pass
            relevant_spans.append(match)
            assign_neutral_label(match, term_mapping)
            all_term_matches.append(match)
    
        # Set the term match variables
        self._num_masc_terms = num_masc_terms
        self._num_fem_terms = num_fem_terms
        self._num_neutral_terms = num_neutral_terms
        self._gendered_term_matches = gendered_term_matches
        self._neutral_term_matches = neutral_term_matches
        
        # Store the results in document spans
        if self._doc.spans.get('glat_term_matches', None) == None:
            # Filter and merge spans to remove overlaps
            merged_spans = spacy.util.filter_spans(relevant_spans)
            # Update the doc's ruler with the final spans
            self._doc.spans["ruler"] = merged_spans 
            self._doc.spans['glat_term_matches'] = all_term_matches
            self._doc.spans['glat'] = [] if (self._doc.spans.get("glat", None) == None) else self._doc.spans['glat']
            for span in all_term_matches:
                self._doc.spans['glat'].append(span)
        
    
    def get_analysis_data(self) -> dict:  
        """
        Retrieves analysis data as a dictionary.

        Returns:
            dict: A dictionary with the following keys:
                - "num_masc_terms" (int): The count of masculine gendered terms found in the text.
                - "num_fem_terms" (int): The count of feminine gendered terms found in the text.
                - "num_neutral_terms" (int): The count of neutral terms found in the text.
                - "excess_pronoun_male" (int): The number of unnecessary masculine pronouns identified.
                - "excess_pronoun_female" (int): The number of unnecessary feminine pronouns identified.
        """
        data = {
                "num_masc_terms": self._num_masc_terms,
                "num_fem_terms": self._num_fem_terms,
                "num_neutral_terms": self._num_neutral_terms, 
                "excess_pronoun_male": self._num_excess_pron_male, 
                "excess_pronoun_female": self._num_excess_pron_female
                }
        return data
    
        
    def display_info(self) -> None:
        """
        Displays the analysis information in table format, including gendered term counts and pronoun usage.
        """
        
        term_table = PrettyTable()
        
        term_table.field_names = ["Term Category", "Number of Matches Found"]
        term_table.add_row(["Masculine", self._num_masc_terms])
        term_table.add_row(["Feminine", self._num_fem_terms])
        term_table.add_row(["Neutral/Ungendered", self._num_neutral_terms], divider=True)
        term_table.add_row(["Gendered", self._num_masc_terms + self._num_fem_terms], divider=True)
        term_table.add_row(["Total", self._num_masc_terms + self._num_fem_terms + self._num_neutral_terms])
        
        pronoun_table = PrettyTable()
        
        pronoun_table.field_names = ["Unnecessarily Gendered Pronouns", "Number of Uses Found"]
        pronoun_table.add_row(["Male (he, him, his, his, himself)", self._num_excess_pron_male])
        pronoun_table.add_row(["Female (she, her, her, hers, herself)", self._num_excess_pron_female])
        
        print(term_table)
        print(pronoun_table, end='\n\n')
        if len(self._gendered_term_matches) > 0:
            print("Gendered Term Matches: ", self._gendered_term_matches, sep='\n', end='\n\n')
        if len(self._neutral_term_matches) > 0:
            print("Gender Neutral Term Matches:", self._neutral_term_matches, sep='\n', end='\n\n')
        if len(self._unnecessary_pronouns['male']) > 0:
            print("Masculine as Default Instances:", self._unnecessary_pronouns['male'], sep='\n', end='\n\n')


        
        
    def visualize_doc(self, key="glat") -> None:
        """
        Visualizes the analyzed document's spans using spaCy's displaCy renderer.

        Args:
            key (str, optional): The span key to visualize. Defaults to "glat".
        """
        key = (key if (key in label_key) else 'glat')
               
        if len(self._doc.spans[key]) == 0:
            print("Visualization can not be done on empty Span.")
            return
        
        colors = {}
        for span in self._doc.spans[key]:
            span_color = ("#00BFFF" if ("masc_" in span.label_.lower()) else "#FFD700" if ("fem_" in span.label_.lower()) else "#eda6a6" if ("female_" in span.label_.lower()) else "#b98cd2" if ("male_" in span.label_.lower()) else "#cae2c2" if ("neutral" in span.label_.lower()) else "")
            if span_color != "":
                colors[span.label_] = span_color 
            
        
        displacy.render(self._doc, style="span", options={"spans_key": key, "colors": colors})
        
            
            
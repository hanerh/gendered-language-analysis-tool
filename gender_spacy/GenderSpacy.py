import spacy
from spacy import displacy
from spacy.tokens import Span, Token
from pathlib import Path

import srsly
from . import components
import toml


pronoun_patterns = "gender_spacy/pronoun_patterns.json"
pronoun_patterns = list(srsly.read_jsonl(pronoun_patterns))

project_toml = "gender_spacy/project.toml"
project_data = toml.load(project_toml)

colors = project_data["colors"]
visualize_params = project_data["visualize_params"]
visualize_params["colors"] = colors

pronouns = project_data["pronouns"]



class GenderSpacy:
    
    def __init__(self) -> None:

        # Initialize extensions for tokens and spans if not already set
        if not Span.has_extension("gt_gender"):
            Span.set_extension("gt_gender", default=None) 
        if not Span.has_extension("gt_linked_pronouns"):
            Span.set_extension("gt_linked_pronouns", default={})
        if not Token.has_extension("gt_gender"):
            Token.set_extension("gt_gender", default=None) 
        if not Token.has_extension("gt_linked_pronouns"):
            Token.set_extension("gt_linked_pronouns", default={})
            
        nlp = spacy.load("en_core_web_sm")
        nlp_coref = spacy.load("en_coreference_web_trf")
        
        # use replace_listeners for the coref components
        nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
        nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

        # we won't copy over the span cleaner
        nlp.add_pipe("coref", source=nlp_coref)
        nlp.add_pipe("span_resolver", source=nlp_coref)
        # nlp.add_pipe("span_cleaner", source=nlp_coref)

        nlp.add_pipe("pronoun_resolver")
        ruler = nlp.add_pipe("span_ruler")
        nlp.add_pipe("pronoun_id")
        nlp.add_pipe('people_and_spouse')
        ruler.add_patterns(pronoun_patterns)
    
        self.nlp = nlp

    def process_doc(self, text):
        """
        Creates the spaCy doc container and iterates over the entities found by the spaCy NER model.
        Args:
            text (str): the text that will be processed.
        
        Returns:
            doc (spaCy Doc): the doc container that contains all the data about gender spans
        """
        doc = self.nlp(text)

        original_spans  = list(doc.spans["ruler"])
        for ent in doc.ents:
            if ent.label_=="PERSON":
                ent.label_ = "PERSON_UNKNOWN"
                original_spans.append(ent)
        doc.spans["ruler"] = original_spans
        self.doc = doc
        return doc
    
    
    def coref_resolution(self):
        """
        Uses the spaCy Experimental Coref Model to identify all connections between PERSON entities and pronouns.
        If there is a cluster where a PERSON entity has a gender-specific pronoun, the span labels are adjusted accordingly.
        """
        spans = list(self.doc.spans["ruler"])
        # pronouns = {"female": ["she", "her", "hers", "herself"], "male": ["he", "him", "his", "himself"]}

        def parse_gender(token, pronoun_set, gender):
            """
            Parse and label tokens based on detected gender pronouns. 
            This function labels spans as PERSON_{GENDER}_COREF.
            """
            if token.text.lower() not in pronoun_set:
                span = self.doc.char_span(token.start_char, token.end_char, label=gender.upper())
                span = Span(self.doc, span.start, span.end, label=f"PERSON_{gender.upper()}_COREF")
                return span
            return None

        # Main loop to iterate over the document's coreference clusters
        for key, cluster in self.doc.spans.items():
            if "head" in key:  # Only work with the coreference "head" clusters
                head_tokens = [token.text.lower() for token in cluster]

                # Check gender for each cluster and pronoun
                for gender, pronoun_set in pronouns.items():
                    # If any pronoun from the set is in the cluster, proceed
                    if any(pronoun in head_tokens for pronoun in pronoun_set):
                        for token in cluster:
                            # Relabel token spans based on gender
                            gender_res = parse_gender(token, pronoun_set, gender.upper())
                            if gender_res:
                                # Extract the cluster number from the key
                                num = key.split("_")[-1]
                                res_cluster = self.doc.spans[f"coref_clusters_{num}"]
                               
                                # Initialize lists for proper nouns, nouns, and pronouns in cluster
                                propn_spans = []
                                pronoun_tokens = {} # To store pronouns linked to the cluster's proper noun
                                noun_spans = []
                                # print(res_cluster)
                                for token_set in res_cluster:
                                    # Add gender extension to the token/cluster entity and relabel pronouns
                                    for token2 in token_set:
                                        # print(token2, token2.pos_)
                                        if token2.pos_ == "PROPN":  # Proper nouns
                                            token2._.gt_gender = gender.upper()  # Add gender extension to the token
                                            # Create a span for the proper noun entity (PERSON)
                                            propn_span = self.doc.char_span(token2.idx, token2.idx + len(token2.text), label=gender.upper())
                                            propn_span = Span(self.doc, propn_span.start, propn_span.end, label=f"PERSON_{gender.upper()}_COREF")
                                            spans.append(propn_span)
                                            propn_spans.append(propn_span) # Collect proper noun for later linking
                                            
                                        elif token2.pos_ == "PRON" and token2.text.lower() in pronoun_set: # Pronouns
                                            token2._.gt_gender = gender.upper()  # Add gender to pronoun
                                            # Collect pronoun for later linking
                                            pron_tok = {token2.i:token2.text}
                                            pronoun_tokens.update(pron_tok) 
                                            
                                        elif token2.pos_ == "NOUN" and len(propn_spans)==0: # Nouns
                                            token2._.gt_gender = gender.upper()  # Add gender extension to the token
                                            noun_span = self.doc.char_span(token2.idx, token2.idx+len(token2.text), label=gender.upper())
                                            noun_span = Span(self.doc, noun_span.start, noun_span.end, label=f"REL_{gender.upper()}_COREF")                  
                                            spans.append(noun_span)
                                            noun_spans.append(noun_span)    # Collect noun for later linking to pronoun if no proper noun in cluster
                                            

                                # After looping through the token_set, if there was a proper noun, link pronouns to it
                                if (len(propn_spans) != 0) and (len(pronoun_tokens) != 0):
                                    # print(f"Linking pronouns {pronoun_tokens} to {propn_spans}")
                                    # Attach pronoun spans to the proper noun's linked_pronouns extension
                                    for ind, text in pronoun_tokens.items():
                                        for propn_span in propn_spans:
                                            propn_span._.gt_linked_pronouns.update({ind: text})
                                            
                                    # Reset for the next iteration
                                    propn_spans = []  # Reset proper noun for the next token_set
                                    pronoun_tokens = {}  # Clear pronoun list for the next token_set
                                
                                # checking for pronouns not attached to proper nouns
                                elif (len(pronoun_tokens) != 0):
                                    # print(f"Linking nouns {pronoun_tokens} to {noun_spans}")
                                    # Attach pronoun spans to the noun's linked_pronouns extension
                                    for ind, text in pronoun_tokens.items():
                                        for noun_span in noun_spans:
                                            noun_span._.gt_linked_pronouns.update({ind: text})
                                            
                                    if gender != "neutral":
                                        #HERE
                                        pass
                                        
                                            
                                    # Reset for the next iteration
                                    noun_spans = []  # Reset proper noun for the next token_set
                                    pronoun_tokens = {}  # Clear pronoun list for the next token_set
             
                                    
        # Now handle the modification of spans and merging them
        def connect_spans(spans):
            """
            Connect adjacent spans if they represent proper nouns that are part of the same entity.
            This merges adjacent proper noun tokens into a larger span.
            """
            new_spans = []
            # print(len(spans))
            for i, span in enumerate(spans):
                # if i % 1000 == 0:
                #     print(i)
                for span2 in spans:
                    if span.end == span2.start:
                        # If both are proper nouns (e.g., first and last names), merge them into one span
                        if (self.doc[span.start].pos_ == "PROPN" and self.doc[span2.start].pos_ == "PROPN") or self.doc[span.start].pos_ == "NOUN" and self.doc[span2.start].pos_ == "NOUN":
                            new_span = Span(self.doc, span.start, span2.end, label=span.label_)
                            if span._.gt_linked_pronouns == span2._.gt_linked_pronouns:
                                new_span._.set("gt_linked_pronouns", span._.gt_linked_pronouns)
                            new_spans.append(new_span)
            spans.extend(new_spans)
            return spans

        # Perform span connection logic
        spans = connect_spans(spans)

        # Final pass: Filter out PERSON_UNKNOWN spans if a known gender span exists at the same start position
        final_spans = []
        for span in spans:
            remove_span = False
            if span.label_ == "PERSON_UNKNOWN":
                # Look for another span with the same start
                for span2 in spans:
                    if span2.start == span.start:
                        remove_span = True
            if not remove_span:
                final_spans.append(span)

        # Filter and merge spans to remove overlaps
        merged_spans = spacy.util.filter_spans(final_spans)

        # Update the doc's ruler with the final spans
        self.doc.spans["ruler"] = merged_spans  

        return self.doc

    def visualize(self, jupyter=True):
        """
        visualizes the spaCy doc Container on the spans
        Args:
            jupyter (Bool): affects if the visualization loads in Jupyter or as HTML
        """
        #########          
        if len(self.doc.spans["ruler"]) == 0:
            print("No gendered pronouns found. Visualization can not be done on empty Span.")
            return
        #########
        
        if jupyter==True:
            displacy.render(self.doc, style="span", options=visualize_params, jupyter=True)
        else:
            displacy.render(self.doc, style="span", options={"spans_key": "ruler"})
            
            

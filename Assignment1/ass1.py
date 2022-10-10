

# Import relevant packages

# Import os to utilize the built in functionality like the current working directory
import os
# For calculations
import numpy as np
# Import pandas to utilize dataframes and to read the xlsx files
import pandas as pd
# For pathnames
import glob
#For unlisting
import itertools
#For using math function like sqrt
import math 
# Find out the current working directory
#os.getcwd()
#os.chdir('/Users/muhkas/Desktop/MK/') #use to set the path to working directory

class Preprocess():
    def __init__(self):
        unique_processed_links= [] 
        
    def loadfile(self, file_path):
        """Read the file and load the data

        Parameters
        ----------
         file_path: str
            A valid file path and file name contianing the data.
            Returns
        ------- 
        Panda Series
            It returns Link column of the file

        """
        df = pd.read_csv(file_path)
        #TODO: Load/read the files and data
        return(df['Link'])

    
    def removeselflinks(self, file_path):
        """Remove the self links and extracts only the outlinks.
           The links are preprocessed to short name eg. https://www.uu.se/contact 
           will be converted to uu.se. 
           A web page can outlink to another page more than once, so duplicates will
           be removed.

        Parameters
        ----------
         file_path: path and name of file to be read
            
       
        """
        #TODO: Call tje laodfile and store retrun values in a variable raw_data
        raw_data = Preprocess.loadfile(self, file_path)
        unique_raw_data = []  
        for i in raw_data:
            if i.find('wikipedia.org')== -1:  #Check if a link is selflink: Files were generated from
                                          #wikipedia, therefore a link contianing 'wikipedia.org'
                                          #represents the inlink and is removed.
                if i.find('/', 8)!=1:  # Check if outlink has long (e-g:https://www.uu.se/contact ) 
                                   # or short (https://www.uu.se) format
                    intermediate_name=i[0: i.find('/', 8)]
                else:
                    intermediate_name=i
             
                if intermediate_name.find('https://')==0:
                    intermediate_name = intermediate_name[8:]
                elif intermediate_name.find('http://')==0:
                    intermediate_name = intermediate_name[7:]
                else:
                    print('error')
        
                if intermediate_name.find('www.')==0:
                    intermediate_name = intermediate_name[4:]
                
                if intermediate_name != [] or intermediate_name!=None: 
                    unique_raw_data.append(intermediate_name)

        unique_raw_data = list(dict.fromkeys(unique_raw_data))
        self.unique_processed_links = unique_raw_data
                #TODO:Remove http:// or https:// etc. and store result in the in variable intermediate_name 
                
                #TODO: #Some addresses are without www. To, keep the same format, www is removed
                #      and store result in the variable intermediate_name
                
                #TODO: #Remove the empty link, if any. 
                #      and append the result in the variable unique_raw_data already defined above for loop.
                
                 
                #TODO: Remove duplicates from variable unique_raw_data and update self.unique_processed_links
                
class PopulateDictionaries(Preprocess):
        def __init__(self):
            self.pages={} # Create a dictionary of pages
            self.pageindex=0 #To keep track of index of the page
            self.inlink_dict={} #a dictionary that record in_links with pagename, pageindex, mainpageindex (page where link originated), and update mainpage indeces, if it was inlinked from more than one pages 
            self.outlink_dict={} #a dictionary that record out_links with pagename, pageindex, outlinkindex (page where link is directed to), and update mainpage indeces, if it was out bound from more than one pages 
            Preprocess.__init__(self)
        
        def addpages(self, list_pages):
            
            """Add pages to a global dictionary of pages and index them 
       

            Parameters
            ----------
             list_pages: list
                A processed list of all pages in a data file
        
            """
            #TODO: Add unique pages and their index in the dictionary pages
            for i,page in enumerate(list_pages):
                if page in self.pages:
                    continue
                else:    
                    self.pages[page] = [len(self.pages)]

        def inlinkgraph(self, out_links):
    
            """Creates dictionary of inlink graph that records in_links with pagename, pageinde and mainpageindex,
                If a webpage is inlinked from more than one main pages then indeces are updated.
                For example: Consider a entery in link_dict is  'usnews.com': [[10], [1], [44]], 
                Then, webpage usnews.com has index 10 and inlinked by main pages 1 and 44.

            Parameters
            ----------
             out_links: list
                A processed list of all pages in a data file
        
          
            """
            for ind, pname in enumerate(out_links):
                #Check if a page already exists in link_dict
                if self.inlink_dict.get(pname)==None: # If a page is not present is dict
                    self.inlink_dict[pname]= [ self.pages[pname], self.pages[out_links[0]] ] # Add the page and indeces
                else: #A page already exists in the inlink_dict, update the main page indeces
                    self.inlink_dict[pname]= [  list(itertools.chain(*self.inlink_dict[pname])), self.pages[out_links[0]]]
            

        def outlinkgraph(self, out_links):
            """Creates dictionary of out link graph that records out_links with pagename, pageindex and mainpageindex,
                If a webpage has many out linkes then indeces are updated.
                For example: Consider a entery in outlink_dict is  'abc.com': [[2], [5], [6]], 
                Then, webpage abc.com has index 2 and outlinked to  pages 5 and 6.

                Parameters
                ----------
                 out_links: list
                    A processed list of all pages in a data file
                
            """
            for ind, pname in enumerate(out_links):
                #TODO: Create outlink dictionary by populating self.outlink_dict see the function description:
                #For example: Consider a entery in outlink_dict is  'abc.com': [[2], [5], [6]], 
                #Then, webpage abc.com has index 2 and outlinked to  pages 5 and 6.
                pindex = self.pages[pname]
                if ind == 0:
                    self.outlink_dict[pname] = [pindex]
                    cur_pagename = pname
                else:
                    self.outlink_dict[cur_pagename].append(pindex)
            #print(f'Current pagename: {cur_pagename}')

from os import link


class AdjacencyMatrices(PopulateDictionaries):
    def __init__(self):
        self.adj_m_pagerank=None #Initialise the adjacancey matrix for pagerank algo.
        self.adj_m_HITS=None    #Initialise the adjacancey matrix for HITS algo.
        PopulateDictionaries.__init__(self) 
    
###############Create Adjacency matrix Page rank
    def adjpagerank(self, dict_inlinks):
        """Adjacacy matrix for page rank algo:  
                        #An Adjacency matrix of in links of web pages divided by total number of out links of a page.
                        #Each element of A that is A_i,j  represents the out link from web page 'i' (row) to web page 'j' (column).
                        #Alternatively,  We can also say that in link from web page 'j' (column) to web page 'i' (row).
                        #Note that  for all 'i' sum(i, M_i,j) = 1 and A must be a square matrix.
          
        Parameters
        ----------
        dict_inlinks : dictionary
               A dictionary of in links
        """
        zero_data = np.zeros(shape=(len(dict_inlinks),len(dict_inlinks)))
        self.adj_m_pagerank = pd.DataFrame(zero_data)
        for i in dict_inlinks:
            link_map=(list(itertools.chain(*dict_inlinks[i])))
            for ind, j in enumerate(link_map):
                if (ind!=0): # It is the page index, but we need both the page index and main page index.
                    #If a page index and main page index is similar then it is self link and is removed.
                    if link_map[ind]!=link_map[0]:
                        self.adj_m_pagerank.iat[link_map[ind], (link_map[0])]=1
        ###########divide the 1 by the total out links (# Only divide if row sum is not 0)
        self.adj_m_pagerank=self.adj_m_pagerank.apply(lambda x : x.div(x.sum()) if (x.sum()!=0) else 0 , axis=1) 
        
        #print(len(self.adj_m_pagerank.axes[0]))
        #print(len(self.adj_m_pagerank.axes[1]))

###############Create Adjacency matrix FOR HITS
    def adjHITS(self, dict_outlinks):
        """Adjacacy matrix for HITS algo:  
                        #An Adjacency matrix of out links of web pages.
                        #Each element of L that is L_i,j  represents the out link from web page 'i' (row) to web page 'j' (column).
                        #Note L must be a square matrix.
                        
        Parameters
        ----------
        dict_inlinks : dictionary
               A dictionary of out links  
        """
        zero_data = np.zeros(shape=(len(dict_outlinks),len(dict_outlinks)))
        self.adj_m_HITS = pd.DataFrame(zero_data)
             #TODO:Populate self.adj_m_HITS as per instructions in the assignment lecture and slides 
        for i in dict_outlinks:
            link_map=(list(itertools.chain(*dict_outlinks[i])))
            for ind, j in enumerate(link_map):
                if ind!=0: # It is the page index, but we need both the page index and main page index.
                    #If a page index and main page index is similar then it is self link and is removed.
                    if link_map[ind]!=link_map[0]:
                        self.adj_m_HITS.iat[ link_map[ind], (link_map[0]) ]=1

class PagerankAlgo():
    def __init__(self, A, d):
        self.d= d       #Teleporting parameter
        
        self.A= A       #An Adjacency matrix of in links of web pages divided by total number of out links of a page.
                        #Each element of A that is A_i,j  represents the out link from web page 'i' (row) to web page 'j' (column).
                        #Note that  for all 'i' sum(i, M_i,j) = 1 and A must be a square matrix.
        self.P= np.ones(len(self.A)) #Intial page rank =1

    def calc_pagerank(self, max_itrs):
        """PageRank Algorithm:  This algorithm was propsed by the Larry Page and Sergey Brin at Stanford University 
                            and it ranks the web pages by measuring their importance.
                            It is used by the search engine Google.
                            
        Parameters
        ----------
        max_itrs : int
               Max number of iterations
    
        Returns
        -------
        numpy array
            A vector of ranks such that p_i is the i-th rank in the range of [0, 1].
    
        Note
        -----
            1) Don't forget to normalize the page rank values in each iteration by max of page rank value.
               This is done to restrict the page rank values in the range of 1-0.
            2) Finally, normalize the page ranks by the sum of values of page ranks. This is only done at the 
               final calculation.
               This is done to so that sum of final page ranks =1.
        """
        #Check if A is square matrix
        assert(self.A.shape[0]==self.A.shape[1])
        #TODO: Implement PageRank algorithm according to assignment lecture and slides.
        page_ranks = np.zeros(len(self.A))
        n = len(self.A)
        self.A = (1-self.d)/n + self.d * self.A
        for i in range(0,max_itrs):
            A_t = self.A.transpose()
            y = A_t @ self.P
            self.P = y/max(y)
            
        page_ranks = self.P/sum(self.P)
        
        return(page_ranks)


class HITSalgo():
    def __init__(self, L):
        self.L= L   #An Adjacency matrix of out links of web pages 
                    #Each element of L that is L_i,j  represents the out link from web page 'i' to web page 'j'.
                    #Note that L must be a square matrix.
                    
        self.a= np.ones(len(L)) #Initial authority values =1
        self.h= np.ones(len(L)) #Initial hub values =1



    def calc_HITS(self, max_itrs):
        """HITS Algorithm:  HITS algorithm was propsed by Jon M. Lleinberg  at Cornell University 
                            and it ranks the web pages by measuring their authoraty and hubs.
                            It is used by the search engine Ask.
        Parameters
        ----------
        max_itrs : int
               Max number of iterations
    
   
        Returns
        -------
        Panda series
            A series conisting of normalized authority and  hub scores.
    
        Note
        -----
        1) Don't forget to normalize the authority score by the sum of sequare values of all authority score. 
        2) Don't forget to normalize the hub score by the sum of sequare values of all hub score.
        """
       
        #Check if adjacency matrix is a square matrix
        assert(self.L.shape[0]==self.L.shape[1])
        
        a_cal=self.a
        h_cal=self.h
        
        #TODO: Implement HITS algorithm according to assignment lecture and slides.
        for i in range(max_itrs):
            a_cal = self.L@self.L.transpose()@a_cal
            a_cal = a_cal/np.sqrt(sum(np.square(a_cal)))

            h_cal = self.L.transpose()@self.L@h_cal
            h_cal = h_cal/(np.sqrt(sum(np.square(h_cal))))

        return(a_cal, h_cal)

#MAIN of the code, #if __name__ == '__main__'
pp_data = AdjacencyMatrices()
file_list =  glob.glob('Data_files' + "/*.csv")
#file_list = ['Data_files/acton.org.csv,', "Data_files/uu.se.csv"]
for fl in file_list:
    pp_data.removeselflinks(fl)
    list_out_links= list(pp_data.unique_processed_links)
    pp_data.addpages(list_out_links)
    pp_data.inlinkgraph(list_out_links)
    pp_data.outlinkgraph(list_out_links)
    pp_data.pageindex += 1

#print(f'Pages: {pp_data.pages}')
#print('In Links: \n', pp_data.inlink_dict)
#print('Out Links:\n', pp_data.outlink_dict ) 
print(' \n')

#AdjacencyMatrices
##.  PageRank
pp_data.adjpagerank(pp_data.inlink_dict)
#print("PageRankAdjMatr \n", pp_data.adj_m_pagerank )
#print( 'If the rows are summing up to one in  adj_m_pagerank:\n',pp_data.adj_m_pagerank.sum(axis=1)[0:25])
#print( 'Number of in-links in adj_m_pagerank: \n',np.count_nonzero(pp_data.adj_m_pagerank, axis=0)[0:25])
#print( 'Number of out-links in adj_m_pagerank: \n',np.count_nonzero(pp_data.adj_m_pagerank, axis=1)[0:25])

##. HITS
pp_data.adjHITS(pp_data.outlink_dict)
#print("HITSAdjMatr \n", pp_data.adj_m_HITS )
# Number of outlinks in adj_m_HITS
#print( 'Number of outlinks in adj_m_HITS:\n',pp_data.adj_m_HITS.sum(axis=1)[0:25])

 

pr= PagerankAlgo(pp_data.adj_m_pagerank , 0.85)
page_rank_score=pr.calc_pagerank(5)
page_rank_score = pd.DataFrame(np.transpose([page_rank_score]), index=pp_data.pages.keys())
page_rank_score.columns = ['PageRank']
print('Highest page rank score is:', max(page_rank_score['PageRank']))
print('Page rank scores:\n', page_rank_score.to_string())

hr= HITSalgo(pp_data.adj_m_HITS )
HITS_scores=hr.calc_HITS(5)
HITS_scores = np.array(HITS_scores)
scores = pd.DataFrame(index=pp_data.pages.keys())
scores.index.name = 'PageName:'
scores['Authority'] = np.transpose(HITS_scores[0])
scores['Hub'] = np.transpose(HITS_scores[1])

print('Highest authority score:', max(HITS_scores[0]))
print('Highest hub:', max( HITS_scores[1]))
print(scores) 
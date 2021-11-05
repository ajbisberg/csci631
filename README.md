### CSCI 631 Class Project

Test dataset: Lending from the 2020 Census
Questions
 - Is this data differentially private? 
 - If we make predicitions using this data, are they fair? 

I think the answer is _no_ to both of these. 
Check out the PDF, 64 features of mostly categorical and continuous variables. 

#### Setup
- create `data/` folder in root directory, store data there. 
- Download zip from `https://www.fhfa.gov/DataTools/Downloads/Pages/Public-Use-Databases.aspx`
  - Scroll down to **Datasets** -> **Single-Family Census Tract File** -> *2020 Fannie Mae Data (zip)*
  - interesting note - Fannie Mae and Freddie Mac are the US Government lenders so they are required to share their loan data publicly. I chose to start with Fannie Mae b/c I think it's bigger? Worth it to check out the other one too.
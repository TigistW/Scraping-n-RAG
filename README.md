# Task 1: PubMed Article Search and PDF Downloader

This project is a **Streamlit application** designed to search for articles on PubMed using a keyword, display the search results, and provide an option to download PDFs (if available). The app is user-friendly and intended to streamline the process of finding and retrieving research articles.

---

## Features

1. **Keyword-based PubMed Search** :

* Input a keyword to search for articles on PubMed.
* Specify the maximum number of articles to retrieve (1–50).

1. **Display Search Results** :

* Shows the PMID, article title, and DOI URL for each article in a well-organized table.

1. **PDF Download** :

* Checks if a downloadable PDF is available for the article.
* Allows you to download the PDF directly via a "Download PDF" button next to the DOI URL.

1. **Dynamic Feedback** :

* Displays confirmation messages when PDFs are successfully downloaded.
* Alerts if a DOI or PDF link is not available.

---

## Project Files

* **`app.py`** :
* Main application file containing the Streamlit code for the UI and functionality.
* **`requirements.txt`** :
* A list of Python dependencies required to run the application.

---

## Prerequisites

1. **Python** :

* Ensure Python 3.8 or higher is installed on your machine.

1. **Dependencies** :

* Install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

---

## Installation & Usage

1. Clone the repository or download the project files:
   ```bash
   git clone https://github.com/repo
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser (it will automatically launch at `http://localhost:8501`).

---

## How to Use

1. **Search for Articles** :

* Enter a keyword in the input box.
* Set the number of articles to retrieve.
* Click **Search PubMed** to fetch the results.

1. **View Results** :

* A table displays the article details: PMID, Title, and DOI URL.
* Each article has a corresponding "Download PDF" button.

1. **Download PDFs** :

* Click the "Download PDF" button next to the DOI URL.
* If a PDF is available, it will be downloaded and saved locally.
* A success message confirms the download.

---

## File Structure

```
pubmed-downloader/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Required Python dependencies
└── README.md            # Project documentation
```

---

## Future Enhancements

1. **Batch PDF Downloads** :

* Add functionality to download all available PDFs at once.

1. **Enhanced PDF Detection** :

* Improve the mechanism for identifying PDF links from article pages.

1. **Export Results** :

* Enable exporting article details (e.g., PMID, title, DOI) to CSV or Excel.

---

## Dependencies

* **Streamlit** : For building the interactive web app.
* **Requests** : For making API calls and fetching data.
* **BeautifulSoup** : For parsing HTML and extracting PDF links.
* **Python 3.8+** : Required for compatibility.

---

## Acknowledgments

* **PubMed API** : For providing access to biomedical literature.
* **Streamlit** : For enabling rapid development of the interactive UI.

I see you've uploaded a file named `Task2.ipynb`. Let me generate a README for this file based on its contents. I'll first review the file to understand the context and purpose of the notebook. Please hold on while I analyze it.

The uploaded notebook contains **31 code cells** and  **6 markdown cells** . Based on the headings in the markdown cells, the primary tasks appear to include:

1. **Libraries** : Setting up the necessary libraries.
2. **Partitioning PDF into text, images, and tables** : Breaking down PDF content into separate components.
3. **Summarize** : Summarizing the extracted information.

Now, I'll draft a README that provides an overview of the notebook's purpose, usage, and features.

# Task 2: PDF Partitioning and Summarization

This Jupyter Notebook is designed to process and analyze PDF documents. It partitions PDFs into their constituent components (text, images, and tables) and summarizes the extracted content for streamlined analysis.

---

## Features

1. **Library Setup** :

* Imports and configures the required libraries for PDF processing, text extraction, and summarization.

1. **PDF Partitioning** :

* Separates PDF content into:
  * **Text** : Extracted textual content from the PDF.
  * **Images** : Visual elements extracted from the PDF.
  * **Tables** : Tabular data extracted and converted into structured formats.

1. **Content Summarization** :

* Generates concise summaries of the extracted text for quick insights.

1. **Versatile Analysis** :

* Processes PDFs of varying formats and complexities, handling both textual and graphical content.

---

## Prerequisites

1. **Python** :

* Ensure Python 3.8 or higher is installed on your system.

1. **Dependencies** :

* Install the necessary Python libraries. A typical `requirements.txt` for this project might include:
  ```plaintext
  requests
  beautifulsoup4
  unstructured[all-docs] 
  pillow 
  lxml
  openai==1.55.0
  langchain 
  langchain-community 
  langchain-openai 
  langchain-groq
  chromadb 
  tiktoken
  python-dotenv 
  protobuf==4.21.5
  httpx==0.27.2
  ```
* Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

---

## How to Use

1. **Load the Notebook** :

* Open the notebook (`Task2.ipynb`) in Jupyter Notebook, JupyterLab, or any compatible IDE.

1. **Run the Cells** :

* Start by running the setup cell under the **Libraries** section to ensure all required packages are imported.

1. **Provide a PDF File** :

* Upload the PDF document you wish to analyze.

1. **Partitioning** :

* Execute the cells in the **Partitioning PDF into text, image, and tables** section to extract and separate the content.

1. **Summarization** :

* Run the summarization cells to generate summaries of the extracted textual content.

---

## Use Cases

1. **Document Analysis** :

* Extract insights from research papers, reports, or books in PDF format.

1. **Content Summarization** :

* Create concise summaries for lengthy documents.

1. **Data Extraction** :

* Extract structured data (tables) for further analysis in tools like Excel or Python's pandas.

1. **Image Analysis** :

* Retrieve visual content for graphic analysis or inclusion in reports.

---

## File Structure

```
Task2/
│
├── Task2.ipynb         # Jupyter Notebook containing the main code
├── sample_pdfs/        # Directory for storing sample PDFs (optional)
└── requirements.txt    # Python dependencies for the project
```

---

## Future Enhancements

1. **Advanced Summarization** :Integrate models like GPT for better summaries of extracted text.
2. **PDF Cleaning** :

* Preprocess PDFs to remove watermarks or irrelevant elements before partitioning.

1. **Batch Processing** :

* Add functionality to process multiple PDFs at once.

1. **Interactive GUI** :

* Build a graphical interface (e.g., using Streamlit) for easier usage.

---

## Dependencies

* **pandas** : For handling tabular data.
* **pdfplumber** : For PDF text and table extraction.
* **pytesseract** : For OCR (optical character recognition) on images.
* **matplotlib** : For visualizing content, if applicable.
* **OpenAI API** : For advanced text summarization, if integrated.

---

Feel free to contribute to this project or report any issues. Happy analyzing! 😊

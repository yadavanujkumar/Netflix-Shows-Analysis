# Netflix Shows Analysis

## Overview
This project analyzes TV shows and movies available on Netflix as of 2019. The dataset is collected from Flixable, a third-party Netflix search engine, and provides insights into Netflix's content strategy and catalog evolution.

## Dataset Information
The dataset (`netflix_titles_nov_2019.csv`) contains information about Netflix content including:
- **show_id**: Unique ID for each show/movie
- **type**: Movie or TV Show
- **title**: Name of the content
- **director**: Director(s)
- **cast**: Cast members
- **country**: Country of production
- **date_added**: Date added to Netflix
- **release_year**: Original release year
- **rating**: Content rating (TV-MA, PG-13, etc.)
- **duration**: Duration (minutes for movies, seasons for TV shows)
- **listed_in**: Genre categories
- **description**: Brief description

## Key Findings & Insights
According to a 2018 report, Netflix has shown interesting trends:
- The number of TV shows has nearly **tripled since 2010**
- The number of movies has **decreased by more than 2,000 titles** since 2010
- This indicates a strategic shift towards TV show content

## Analysis Questions
This project explores several interesting questions:

1. **Content Distribution by Country**: Understanding what content is available in different countries
2. **Content Similarity Analysis**: Identifying similar content by matching text-based features (descriptions, genres)
3. **Network Analysis**: Analyzing relationships between Actors and Directors to find interesting insights
4. **TV vs Movies Trend**: Investigating if Netflix is increasingly focusing on TV rather than movies in recent years
5. **Genre Popularity**: What genres are most common on Netflix
6. **Content Rating Distribution**: Understanding the target audience through content ratings

## Project Structure
```
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── netflix_titles_nov_2019.csv    # Dataset
├── requirements.txt               # Python dependencies
├── netflix_analysis.py            # Main analysis script
└── .gitignore                     # Git ignore patterns
```

## Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
1. Clone the repository
```bash
git clone https://github.com/yadavanujkumar/Netflix-Shows-Analysis.git
cd Netflix-Shows-Analysis
```

2. Install required packages
```bash
pip install -r requirements.txt
```

### Usage
Run the analysis script:
```bash
python netflix_analysis.py
```

This will generate various visualizations and insights about the Netflix dataset.

## Analysis Components

### 1. Data Exploration
- Basic statistics about the dataset
- Missing data analysis
- Data type validation

### 2. Content by Country
- Visualization of content distribution across countries
- Top countries producing Netflix content

### 3. TV Shows vs Movies Trend
- Year-over-year analysis of content type
- Visualization of the shift from movies to TV shows

### 4. Genre Analysis
- Most popular genres on Netflix
- Genre distribution for movies vs TV shows

### 5. Content Similarity
- Text-based similarity using TF-IDF on descriptions
- Recommendations based on similar content

### 6. Network Analysis
- Actor and Director collaboration networks
- Most prolific actors and directors

### 7. Predictive Modeling
- Machine learning models to predict content characteristics
- Content type prediction (Movie vs TV Show)
- Content rating prediction based on descriptions and genres
- Release decade prediction from content features
- Model evaluation with accuracy metrics and feature importance

## Future Enhancements
- Integration with IMDB ratings dataset for quality analysis
- Integration with Rotten Tomatoes for critic vs audience reception
- Sentiment analysis on content descriptions
- Time series analysis of content addition patterns
- Advanced deep learning models for content success prediction

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset source: [Flixable](https://flixable.com/)
- Inspired by Netflix's content strategy evolution
- Community contributions and insights
# Norwegian Sentiment Analysis

This project focuses on document-level sentiment analysis of Norwegian texts, comparing three different models. The models are traind on the Norwegian Review Corpus, [NoReC](https://github.com/ltgoslo/norec), which has been made specifically for the purpose of training and evaluating models for document-level sentiment analysis. It's comprised of ~43K reviews collected from eight different sources in a number of different domains.

## Results

<table>
<thead>
  <tr>
    <th>Classification Type</th>
    <th>Model</th>
    <th>AUC (%)</th>
    <th>Accuracy (%)</th>
    <th>F1 (%)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Binary</td>
    <td>nb-bert</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>norbert3</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>84.89</td>
    <td>83.36</td>
    <td>89.23</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>73.56</td>
    <td>85.10</td>
    <td>90.77</td>
  </tr>
  <tr>
    <td rowspan="4">Multiclass</td>
    <td>nb-bert</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>norbert3</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>80.52</td>
    <td>63.09</td>
    <td>63.08</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>81.65</td>
    <td>65.18</td>
    <td>64.61</td>
  </tr>
</tbody>
</table>
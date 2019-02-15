# Invoice selection for Auction

## Usage

<ul>
<li>> git clone [repositoru-url]</li>
<li>Launch "Invoice Selection.ipynb" with jupyter notebook</li>
<li>Read the test dataset as a pandas DataFrame</li>
<li>NOTE:Test dataset must have:[ 'Invoice Value' , 'Invoice Load Date in PV' , 'Original Invoice Pay Schedule Date' ]</li>
<li><b>Model supports prediction on invoices with Supplier info ([ 'SUPP_ID' , 'Supplier_Category']) and w/o supplier info</b></li>
<li>Prediction Accuracy on Validation set with Supplier info: 99.98%</li>
<li>Prediction Accuracy on Validation set w/o Supplier info: 76%</li>
</ul> 

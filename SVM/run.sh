echo PrimalSVM schedule1
python3 SVM_primal_domain.py primal_gamma1

echo PrimalSVM schedule2
python3 SVM_primal_domain.py primal_gamma2

echo Dual SVM
python3 dual_SVM.py

echo kernel perceptron
python3 kernal_perceptron.py

#nohup python score.py  & 
# sudo nohup docker run --publish 7878:5000 python-docker  &
export FLASK_APP=P7
flask run  --port=4567 --host=51.158.147.66
#sudo -b docker run --publish 7878:5000 python-docker

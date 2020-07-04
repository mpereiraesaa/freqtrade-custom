#Deploy these files by SSH over aws instance.
scp -i ~/.ssh/myamazon.pem ./config.json ec2-user@ec2-3-128-27-237.us-east-2.compute.amazonaws.com:~
scp -i ~/.ssh/myamazon.pem ./user_data/strategies/initial_strategy.py ec2-user@ec2-3-128-27-237.us-east-2.compute.amazonaws.com:~
FROM	node:16

WORKDIR	/app
ADD    ./package.json /app/package.json
ADD    ./package-lock.json /app/package-lock.json

RUN	npm install

ADD	. /app

EXPOSE	3000
CMD	npm start
# Frontend Dockerfile | Documentation

- [Frontend Dockerfile | Documentation](#frontend-dockerfile--documentation)
  - [Build Stage](#build-stage)
    - [Use an official Node.js image](#use-an-official-nodejs-image)
    - [Install dependencies and build the project](#install-dependencies-and-build-the-project)
  - [Image creation stage](#image-creation-stage)
    - [Use a lightweight web server for serving static files](#use-a-lightweight-web-server-for-serving-static-files)
    - [Copy the dist folder to the nginx image](#copy-the-dist-folder-to-the-nginx-image)
    - [Make port 80 available to the world outside this container](#make-port-80-available-to-the-world-outside-this-container)
    - [Run nginx when the container launches](#run-nginx-when-the-container-launches)

## Build Stage

### Use an official Node.js image

```dockerfile
FROM node:current AS builder
WORKDIR /app
```

### Install dependencies and build the project

```dockerfile
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
```

At this point, we've generated a `dist` folder which contains all the
static files we need to serve the frontend application.

## Image creation stage

### Use a lightweight web server for serving static files

```dockerfile
FROM nginx:alpine
```

HUGE! We actually don't use the builder stage at all, it's only needed for generating the dist folder The dist folder is copied to the nginx image, which is the final image. Wow! The dist folder looks like this:

- assets
  - index-XXXXXXXX.js
  - index-XXXXXXXX.css
  - index-XXXXXXXX.svg
- index.html
- favicon.ico

### Copy the dist folder to the nginx image

```dockerfile
COPY --from=builder /app/dist /usr/share/nginx/html
```

### Make port 80 available to the world outside this container

NOTE: Expose actually does nothing, it's just a hint for the user.
You still need to specify port like: `docker run -p 80:80 my-image`

```dockerfile
EXPOSE 80
```

### Run nginx when the container launches

If we do not launch with `daemon off`, the CMD command will be executed in detached mode which doesn't even take a second to complete. Resulting behavior is that nginx will start and the docker container will exit immediately. We want the container to keep running.

```dockerfile
CMD ["nginx", "-g", "daemon off;"]
```

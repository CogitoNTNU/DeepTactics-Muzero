# Docker Compose

The docker compose tool enables us to streamline the process of building, running, and managing multi-container Docker applications. When using Docker Compose, you don't need to go into the `backend` and `frontend` folders and build using `docker build` and run using `docker run` commands. Instead, you can define the services in a `docker-compose.yml` file and run `docker-compose up` to build and run the services.

## Services

### Backend Service

```yaml
services:
  backend:
    build: ./backend
    image: muzero-backend
    expose:
      - "8080"
    volumes:
     - ./backend:/app 
    networks:
      - muzero-network
```

#### Volumes property

The `./backend:/app` setting is used to Set up a shared volume between the host and the container. Changes you make to the files in the backend folder will be reflected in the container. Thus, you don't need to rebuild the container every time you make a change in the host backend folder. Similarly, if the container writes to the /app folder, the changes will be reflected in the host backend folder.

### Frontend Service

```yaml
  frontend:
    build: ./frontend
    image: muzero-frontend
    ports:
      - "9135:80"
    depends_on:
      - backend
    networks:
      - muzero-network
```

### Network

```yaml
networks:
  muzero-network:
    driver: bridge
```

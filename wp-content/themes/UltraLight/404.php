<?php
/*
Template Name: Error 404
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2>Error 404</h2>
<?php while (have_posts()) : the_post(); ?>
			<div class="page">
				<p>Beklager, men siden du leter etter ser ikke ut til å eksistere</p>
			</div>
<?php endwhile; ?>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>

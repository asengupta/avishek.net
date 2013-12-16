<?php
/*
Template Name: Page
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2><?php the_title(); ?></h2>

<?php while (have_posts()) : the_post(); ?>
			<div class="page">
				<?php the_content(''); ?>
			</div>
<?php endwhile; ?>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>
